#!/usr/bin/env python3
"""Run the multi-card sweep: reng-tp over the roster at every admissible world,
and N data-parallel replicas of reng-bench, into the same JSON shape the
single-card benches write plus a world and a strategy per entry.

usage: sweep_tp.py <reng-tp> <reng-bench> <out_prefix>
                   [--modules 0,1,2,3,4,5,6,7] [--worlds 2,4,8]
                   [--replicas 1,2,4,8] [--batches 1,8] [--prompt 128]
                   [--new 64] [--capacity 1024] [--ids "1 2 3 4 5"]
                   [--prompt-dir <dir>] [--ceiling <reng-ceiling>] [--ctx 192]
                   [--timeout 2900] [--strategies tp,dp] [--repeats 3]
                   <model_dir> [<model_dir> ...]

This needs every card named in --modules, so it is not part of the bench
workflow's default path: run it by hand, or through the workflow's
`multi-card sweep` job, which is workflow_dispatch only.

A tensor-parallel cell is one reng-tp coordinator over the first `world`
modules; the world is skipped, with the reason recorded, when the Megatron
split does not divide the shape (num_attention_heads, num_key_value_heads and
intermediate_size must all divide the world). A data-parallel cell is `n`
reng-bench processes started together, one per module, and its value is the
sum of their throughputs. Percentages are against the strategy's own ceiling
from `reng-ceiling --json`, when that binary is passed.

Every cell is `--repeats` invocations (3 by default, the number the README's
table is built from) and its value is the median of them: at world 8 one run
in five stalls on a collective, so a single shot can be off by more than an
order of magnitude. Every repeat is kept in the `-cells.json` file. A failed or
timed-out cell does not stop the sweep; it is recorded as failed.
"""
import json
import os
import re
import subprocess
import sys


def parse_args(argv):
    if len(argv) < 4:
        sys.exit(__doc__)
    tp, bench, prefix = argv[0], argv[1], argv[2]
    opts = {"modules": "0,1,2,3,4,5,6,7", "worlds": "2,4,8", "replicas": "1,2,4,8",
            "batches": "1,8", "prompt": "128", "new": "64", "capacity": "1024",
            "ids": "1 2 3 4 5", "prompt-dir": "", "ceiling": "", "ctx": "192",
            "timeout": "2900", "strategies": "tp,dp", "repeats": "3"}
    models = []
    i = 3
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            key = a[2:]
            if key not in opts:
                sys.exit(f"unknown option {a}; known: " + ", ".join(sorted(opts)))
            if i + 1 >= len(argv):
                sys.exit(f"{a} needs a value")
            opts[key] = argv[i + 1]
            i += 2
        else:
            models.append(a)
            i += 1
    if not models:
        sys.exit("no model directory given")
    for key in ("prompt", "new", "capacity", "ctx", "timeout", "repeats"):
        if not opts[key].isdigit() or int(opts[key]) < 1:
            sys.exit(f"--{key} wants a positive integer, got {opts[key]!r}")
    return tp, bench, prefix, opts, models


def median_cell(cells):
    """The median of the repeats by tok/s, with every repeat kept beside it."""
    ok = [c for c in cells if c["ok"]]
    if not ok:
        out = dict(cells[0])
        out["repeats"] = cells
        return out
    ok.sort(key=lambda c: c["tok_s"])
    out = dict(ok[len(ok) // 2])
    out["repeats"] = cells
    out["tok_s_range"] = [ok[0]["tok_s"], ok[-1]["tok_s"]]
    return out


def config(model):
    with open(os.path.join(model, "config.json")) as f:
        return json.load(f)


def split_reason(cfg, world):
    """Why LlamaConfig::shard would reject this world, or None."""
    heads = cfg.get("num_attention_heads", 0)
    kv = cfg.get("num_key_value_heads", heads)
    ff = cfg.get("intermediate_size", 0)
    bad = []
    if heads % world:
        bad.append(f"{heads} heads")
    if kv % world:
        bad.append(f"{kv} kv heads")
    if ff % world:
        bad.append(f"intermediate {ff}")
    return f"{' and '.join(bad)} not divisible by {world}" if bad else None


def ceilings(opts, model, cards, batch):
    """reng-ceiling's plan table for this model, cards and batch."""
    if not opts["ceiling"]:
        return None
    cmd = [opts["ceiling"], model, "--cards", str(cards), "--batch", str(batch),
           "--ctx", opts["ctx"], "--json"]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        print(f"  reng-ceiling failed: {r.stderr.strip()[:200]}", flush=True)
        return None
    return json.loads(r.stdout)


def plan(table, strategy):
    if not table:
        return None
    for p in table["plans"]:
        if p["strategy"] == strategy:
            return p
    return None


def run_tp(tp, model, world, batch, opts, out):
    """One tensor-parallel cell: one coordinator over `world` modules."""
    mods = ",".join(opts["modules"].split(",")[:world])
    cmd = [tp, model, opts["new"], "--modules", mods, "--batch", str(batch),
           "--bench", opts["new"], "--capacity", opts["capacity"],
           "--out", out, "--timeout", opts["timeout"]]
    name = os.path.basename(model.rstrip("/"))
    pf = os.path.join(opts["prompt-dir"], f"{name}-{opts['prompt']}.json") if opts["prompt-dir"] else ""
    if pf and os.path.exists(pf):
        cmd += ["--prompt-file", pf]
    else:
        cmd += opts["ids"].split()
    # `--timeout` is reng-tp's own deadline; the wall clock here is a
    # backstop for a coordinator that never returns at all, so that one hung
    # cell cannot hold every card for the rest of the sweep.
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=False,
                           timeout=int(opts["timeout"]) + 120)
    except subprocess.TimeoutExpired:
        return {"ok": False, "log": f"timed out after {int(opts['timeout']) + 120} s"}
    # The coordinator prefixes each rank's line with "[r0] ".
    m = re.search(r"RESULT: decode b\d+ ([0-9.]+) tok/s \(([0-9.]+) ms/step\)", r.stdout)
    if r.returncode != 0 or not m:
        return {"ok": False, "log": (r.stdout + r.stderr)[-400:]}
    cell = {"ok": True, "tok_s": float(m.group(1)), "step_ms": float(m.group(2))}
    ar = re.search(r"all-reduces ([0-9.]+) us", r.stdout)
    if ar:
        cell["all_reduce_us_per_layer"] = float(ar.group(1))
    return cell


def run_dp(bench, model, replicas, batch, opts):
    """One data-parallel cell: `replicas` reng-bench processes, one per module,
    started together and waited on together. The value is their sum."""
    mods = opts["modules"].split(",")[:replicas]
    name = os.path.basename(model.rstrip("/"))
    procs = []
    for m in mods:
        env = dict(os.environ, RENG_MODULE_ID=m)
        out = f"/tmp/sweep-tp-{name}-dp{replicas}-b{batch}-m{m}.json"
        cmd = [bench, model, out, "--prompt", opts["prompt"], "--new", opts["new"],
               "--capacity", opts["capacity"], "--batch", str(batch), "--warmup", "1"]
        procs.append((m, subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                          stderr=subprocess.STDOUT, text=True)))
    per = []
    logs = []
    deadline = int(opts["timeout"]) + 120
    for m, p in procs:
        try:
            log = p.communicate(timeout=deadline)[0]
        except subprocess.TimeoutExpired:
            # One hung replica would otherwise hold every card forever.
            p.kill()
            log = (p.communicate()[0] or "") + f"\nkilled after {deadline} s"
        logs.append(f"m{m}: {log[-200:]}")
        hit = re.search(r"^decode .*?: ([0-9.]+) tok/s, \w+ step ([0-9.]+) ms", log, re.M)
        if p.returncode != 0 or not hit:
            per.append(None)
        else:
            per.append((float(hit.group(1)), float(hit.group(2))))
    if any(x is None for x in per):
        return {"ok": False, "log": "\n".join(logs)[-400:]}
    return {"ok": True, "tok_s": sum(x[0] for x in per),
            "step_ms": max(x[1] for x in per),
            "per_process_tok_s": [x[0] for x in per]}


def entries(name, cell, strategy, world, batch, ceiling_tok_s, extra):
    """The single-card bench JSON shape, plus world and strategy. `cell` is a
    median over the repeats, as the README's table is."""
    out = [{"name": f"{name} decode tok/s (b{batch}) {strategy} n{world}",
            "unit": "tok/s", "value": cell["tok_s"], "extra": extra,
            "world": world, "strategy": strategy}]
    if ceiling_tok_s:
        out.append({"name": f"{name} decode % of ceiling (b{batch}) {strategy} n{world}",
                    "unit": "%", "value": 100.0 * cell["tok_s"] / ceiling_tok_s,
                    "extra": f"ceiling {ceiling_tok_s:.1f} tok/s",
                    "world": world, "strategy": strategy})
    return out


def main():
    tp, bench, prefix, opts, models = parse_args(sys.argv[1:])
    modules = opts["modules"].split(",")
    worlds = [int(w) for w in opts["worlds"].split(",") if w]
    replicas = [int(n) for n in opts["replicas"].split(",") if n]
    batches = [int(b) for b in opts["batches"].split(",") if b]
    want = opts["strategies"].split(",")
    rows, out_entries = {}, []

    for model in models:
        name = os.path.basename(model.rstrip("/"))
        cfg = config(model)
        rows[name] = {}
        for batch in batches:
            if "tp" in want:
                for world in worlds:
                    key = f"tp n{world} b{batch}"
                    if world > len(modules):
                        why = f"only {len(modules)} modules"
                        rows[name][key] = {"ok": False, "skipped": why}
                        print(f"{name} {key}: skipped, {why}", flush=True)
                        continue
                    why = split_reason(cfg, world)
                    if why:
                        rows[name][key] = {"ok": False, "skipped": why}
                        print(f"{name} {key}: skipped, {why}", flush=True)
                        continue
                    cell = median_cell([
                        run_tp(tp, model, world, batch, opts,
                               f"/tmp/sweep-tp-{name}-w{world}-b{batch}-r{r}.json")
                        for r in range(1, int(opts["repeats"]) + 1)
                    ])
                    rows[name][key] = cell
                    table = ceilings(opts, model, world, batch)
                    p = plan(table, "tensor")
                    ceil = p["aggregate_tok_s"] if p and batch > 1 else (
                        p["single_stream_tok_s"] if p else None)
                    if cell["ok"]:
                        out_entries += entries(
                            name, cell, "tensor", world, batch, ceil,
                            f"{cell['step_ms']:.2f} ms/step at world {world}"
                            + (f"; ceiling {ceil:.1f} tok/s practical" if ceil else ""))
                        pct = f" ({100.0 * cell['tok_s'] / ceil:.1f}%)" if ceil else ""
                        rng = cell.get("tok_s_range", [])
                        spread = (f" [{rng[0]:.1f}-{rng[1]:.1f} over "
                                  f"{opts['repeats']}]") if rng else ""
                        print(f"{name} {key}: {cell['tok_s']:.1f} tok/s{pct}{spread}",
                              flush=True)
                    else:
                        print(f"{name} {key}: failed", flush=True)
            if "dp" in want:
                for n in replicas:
                    key = f"dp n{n} b{batch}"
                    if n > len(modules):
                        rows[name][key] = {"ok": False, "skipped": f"only {len(modules)} modules"}
                        continue
                    cell = median_cell([run_dp(bench, model, n, batch, opts)
                                        for _ in range(int(opts["repeats"]))])
                    rows[name][key] = cell
                    table = ceilings(opts, model, n, batch)
                    p = plan(table, "data")
                    ceil = p["aggregate_tok_s"] if p else None
                    if cell["ok"]:
                        out_entries += entries(
                            name, cell, "data", n, batch, ceil,
                            f"{n} replicas, aggregate; slowest step {cell['step_ms']:.2f} ms"
                            + (f"; ceiling {ceil:.1f} tok/s" if ceil else ""))
                        pct = f" ({100.0 * cell['tok_s'] / ceil:.1f}%)" if ceil else ""
                        print(f"{name} {key}: {cell['tok_s']:.1f} tok/s aggregate{pct}", flush=True)
                    else:
                        print(f"{name} {key}: failed", flush=True)

    with open(prefix + ".json", "w") as f:
        json.dump(out_entries, f, indent=2)
    with open(prefix + "-cells.json", "w") as f:
        json.dump({"options": opts, "results": rows}, f, indent=2)

    columns = []
    for batch in batches:
        columns += [f"tp n{w} b{batch}" for w in worlds if "tp" in want]
        columns += [f"dp n{n} b{batch}" for n in replicas if "dp" in want]
    lines = ["| Model | " + " | ".join(f"{c} tok/s" for c in columns) + " |",
             "|---|" + "---|" * len(columns)]
    for name, cells in rows.items():
        row = []
        for c in columns:
            cell = cells.get(c)
            if cell is None:
                row.append("-")
            elif cell["ok"]:
                row.append(f"{cell['tok_s']:.0f}")
            elif cell.get("skipped"):
                row.append(cell["skipped"])
            else:
                row.append("failed")
        lines.append(f"| {name} | " + " | ".join(row) + " |")
    with open(prefix + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
