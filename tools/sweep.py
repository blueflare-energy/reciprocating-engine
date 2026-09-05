#!/usr/bin/env python3
"""Run reng-bench over a grid of models and batch sizes and write the
results as a Markdown table plus JSON (the plan's Chart 2: decode throughput
and percent of the roofline ceiling per model and batch).

usage: sweep.py <reng-bench> <out_prefix> [--batches 1,8,64] [--prompt 128]
                [--new 32] [--capacity 1024] <model_dir> [<model_dir> ...]

Runs on the Gaudi host; HABANA_VISIBLE_DEVICES selects the card. Every
(model, batch) cell is one reng-bench invocation, so a failed cell does not
stop the sweep; it is recorded as such.
"""
import json
import os
import re
import subprocess
import sys


def parse_args(argv):
    if len(argv) < 3:
        sys.exit(__doc__)
    bench, prefix = argv[0], argv[1]
    opts = {"batches": "1,8,64", "prompt": "128", "new": "32", "capacity": "1024"}
    models = []
    i = 2
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            opts[a[2:]] = argv[i + 1]
            i += 2
        else:
            models.append(a)
            i += 1
    return bench, prefix, opts, models


def run_cell(bench, model, batch, opts):
    out = f"/tmp/sweep-{os.path.basename(model)}-b{batch}.json"
    cmd = [bench, model, out, "--prompt", opts["prompt"], "--new", opts["new"],
           "--capacity", opts["capacity"], "--batch", str(batch), "--warmup", "1"]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    m = re.search(r"^decode .*?: ([0-9.]+) tok/s, median step ([0-9.]+) ms.*= ([0-9.]+)%",
                  r.stdout, re.M)
    if r.returncode != 0 or not m:
        return {"ok": False, "log": (r.stdout + r.stderr)[-400:]}
    return {"ok": True, "tok_s": float(m.group(1)), "step_ms": float(m.group(2)),
            "pct": float(m.group(3))}


def main():
    bench, prefix, opts, models = parse_args(sys.argv[1:])
    batches = [int(b) for b in opts["batches"].split(",")]
    table = {}
    for model in models:
        name = os.path.basename(model.rstrip("/"))
        table[name] = {}
        for b in batches:
            cell = run_cell(bench, model, b, opts)
            table[name][b] = cell
            status = f"{cell['tok_s']:.0f} tok/s ({cell['pct']:.1f}%)" if cell["ok"] else "failed"
            print(f"{name} b{b}: {status}", flush=True)
    with open(prefix + ".json", "w") as f:
        json.dump({"options": opts, "results": table}, f, indent=2)
    lines = ["| Model | " + " | ".join(f"b{b} tok/s (% ceiling)" for b in batches) + " |",
             "|---|" + "---|" * len(batches)]
    for name, cells in table.items():
        row = [f"{c['tok_s']:.0f} ({c['pct']:.1f}%)" if c["ok"] else "failed"
               for c in (cells[b] for b in batches)]
        lines.append(f"| {name} | " + " | ".join(row) + " |")
    with open(prefix + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
