"""transformers reference for the engine's RoPE tables (CPU, f32).

usage: rope_reference.py <out.json> [--models-dir DIR] [--offline]

Writes the fixture `crates/reng-model/testdata/rope_reference.json` reads:
for every case the checkpoint's own `config.json`, the inverse frequencies
and attention factor transformers derives from it
(`modeling_rope_utils.ROPE_INIT_FUNCTIONS`), and the `cos` / `sin` rows the
model's rotary module returns at a handful of positions on both sides of the
pretraining length. No weights are loaded and nothing runs on a device: the
rotary module is instantiated from the config alone.

The cases are the three checkpoints whose scaling types the engine gained
(Phi-3.5-mini-instruct and Phi-4-mini-instruct: longrope, the second with a
partial rotation; gemma-3-4b: a multimodal config whose text half scales
`linear`) plus three synthetic yarn configs, which no roster checkpoint
carries. A model directory that holds no `config.json` is filled from the
Hub with

    hf download <repo> --include config.json --local-dir <models-dir>/<name>

`--offline` skips that and the case with it.

A longrope module picks its long factors when `max(position_ids) + 1`
exceeds `original_max_position_embeddings`, so a case's positions decide
which table it reports; `keep` drops the trailing position that only pushed
the module over the threshold, which is how the engine's `seq_len` argument
is exercised apart from the table length.
"""

import json
import os
import subprocess
import sys

import torch
from transformers import AutoConfig, LlamaConfig

MODELS = os.path.expanduser("~/models")
TOKEN = os.environ.get("HF_TOKEN", "")

# (name, repo, model directory, engine table, layer type, positions, keep)
CHECKPOINTS = [
    ("phi35-short", "microsoft/Phi-3.5-mini-instruct", "Phi-3.5-mini-instruct",
     "global", None, [0, 1, 7, 300, 2000, 4095], None),
    ("phi35-long", "microsoft/Phi-3.5-mini-instruct", "Phi-3.5-mini-instruct",
     "global", None, [0, 1, 300, 4096, 4097, 4500], None),
    ("phi35-long-short-table", "microsoft/Phi-3.5-mini-instruct", "Phi-3.5-mini-instruct",
     "global", None, [0, 1, 7, 300, 4096], 4),
    ("phi4mini-short", "microsoft/Phi-4-mini-instruct", "Phi-4-mini-instruct",
     "global", None, [0, 1, 7, 300, 2000, 4095], None),
    ("phi4mini-long", "microsoft/Phi-4-mini-instruct", "Phi-4-mini-instruct",
     "global", None, [0, 1, 300, 4096, 4097, 4500], None),
    ("gemma3-4b-global", "google/gemma-3-4b-pt", "Gemma-3-4B",
     "global", "full_attention", [0, 1, 7, 300, 1024, 2000], None),
    ("gemma3-4b-local", "google/gemma-3-4b-pt", "Gemma-3-4B",
     "local", "sliding_attention", [0, 1, 7, 300, 1024, 2000], None),
]

# Synthetic yarn configs: the shape of a DeepSeek-V2-Lite / V4 or Kimi-K2
# rotary table (64 rotary dims) on a llama config the engine also parses.
YARN_BASE = {
    "model_type": "llama", "architectures": ["LlamaForCausalLM"],
    "hidden_size": 512, "intermediate_size": 1024, "num_hidden_layers": 2,
    "num_attention_heads": 8, "num_key_value_heads": 8, "head_dim": 64,
    "rms_norm_eps": 1e-6, "vocab_size": 128, "tie_word_embeddings": True,
}
YARN = [
    ("yarn-dsv2lite", 10000.0, 163840, {
        "rope_type": "yarn", "factor": 40.0, "original_max_position_embeddings": 4096,
        "beta_fast": 32, "beta_slow": 1, "mscale": 0.707, "mscale_all_dim": 0.707}),
    ("yarn-kimi-k2", 50000.0, 131072, {
        "rope_type": "yarn", "factor": 32.0, "original_max_position_embeddings": 4096,
        "beta_fast": 1.0, "beta_slow": 1.0, "mscale": 1.0, "mscale_all_dim": 1.0}),
    ("yarn-dsv4-flash", 10000.0, 1048576, {
        "rope_type": "yarn", "factor": 16.0, "original_max_position_embeddings": 65536,
        "beta_fast": 32, "beta_slow": 1}),
]
YARN_POSITIONS = [0, 1, 7, 300, 4096]


def fetch_config(repo, path):
    """Download the repo's config.json into `path` when it is missing."""
    if os.path.exists(os.path.join(path, "config.json")):
        return True
    hf = os.path.join(os.path.dirname(sys.executable), "hf")
    cmd = [hf if os.path.exists(hf) else "hf",
           "download", repo, "--include", "config.json", "--local-dir", path]
    env = dict(os.environ, HF_TOKEN=TOKEN) if TOKEN else os.environ
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, env=env, check=False).returncode == 0


def rotary_module(cfg):
    """The checkpoint's own rotary module, and whether it takes a layer type."""
    if cfg.model_type == "phi3":
        from transformers.models.phi3.modeling_phi3 import Phi3RotaryEmbedding
        return Phi3RotaryEmbedding(cfg), False
    if cfg.model_type in ("gemma3", "gemma3_text"):
        from transformers.models.gemma3.modeling_gemma3 import Gemma3RotaryEmbedding
        text = cfg.text_config if cfg.model_type == "gemma3" else cfg
        return Gemma3RotaryEmbedding(text), True
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
    return LlamaRotaryEmbedding(cfg), False


def tables(cfg, layer_type, positions):
    """(inv_freq, attention_factor, cos, sin) at `positions`, as HF computes them."""
    mod, per_layer = rotary_module(cfg)
    pos = torch.tensor([positions], dtype=torch.long)
    x = torch.zeros(1, 1, 1, dtype=torch.float32)
    if per_layer:
        cos, sin = mod(x, pos, layer_type=layer_type)
        inv = getattr(mod, f"{layer_type}_inv_freq")
        af = getattr(mod, f"{layer_type}_attention_scaling")
    else:
        cos, sin = mod(x, pos)
        inv, af = mod.inv_freq, mod.attention_scaling
    return inv, float(af), cos[0], sin[0]


def rows(t):
    """One `cos` / `sin` row per position, `[positions, rotary_dim]` f32."""
    out = []
    for row in t:
        v = row.tolist()
        half = len(v) // 2
        assert v[:half] == v[half:], "cos/sin rows repeat their half"
        out.append([float(f"{x:.9g}") for x in v])
    return out


def case(name, cfg, raw, layer_type, table, positions, keep):
    inv, af, cos, sin = tables(cfg, layer_type, positions)
    kept = len(positions) if keep is None else keep
    text = cfg.text_config if cfg.model_type == "gemma3" else cfg
    head_dim = getattr(text, "head_dim", None) or text.hidden_size // text.num_attention_heads
    prf = getattr(text, "partial_rotary_factor", None) or 1.0
    out = {
        "name": name,
        "config": raw,
        "table": table,
        "layer_type": layer_type,
        "head_dim": head_dim,
        "rotary_dim": int(head_dim * prf),
        "seq_len": max(positions) + 1,
        "table_len": max(positions[:kept]) + 1,
        "attention_factor": af,
        "inv_freq": [float(f"{x:.9g}") for x in inv.tolist()],
        "positions": positions[:kept],
        "cos": rows(cos[:kept]),
        "sin": rows(sin[:kept]),
    }
    assert len(out["inv_freq"]) * 2 == out["rotary_dim"], name
    assert len(out["cos"][0]) == out["rotary_dim"], name
    print(f"{name}: rotary_dim {out['rotary_dim']}, seq_len {out['seq_len']}, "
          f"table_len {out['table_len']}, attention_factor {af:.9g}, "
          f"inv_freq[0] {out['inv_freq'][0]:.9g}, inv_freq[-1] {out['inv_freq'][-1]:.9g}",
          flush=True)
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    offline = "--offline" in sys.argv
    models = MODELS
    if "--models-dir" in sys.argv:
        models = sys.argv[sys.argv.index("--models-dir") + 1]
    out = args[0] if args else "rope_reference.json"

    cases = []
    for name, repo, dirname, table, layer_type, positions, keep in CHECKPOINTS:
        path = os.path.join(models, dirname)
        if not offline and not fetch_config(repo, path):
            print(f"{name}: no config.json ({repo}), skipped", flush=True)
            continue
        if not os.path.exists(os.path.join(path, "config.json")):
            print(f"{name}: no config.json in {path}, skipped", flush=True)
            continue
        raw = json.load(open(os.path.join(path, "config.json")))
        cfg = AutoConfig.from_pretrained(path)
        cases.append(case(name, cfg, raw, layer_type, table, positions, keep))

    for name, theta, max_pos, scaling in YARN:
        raw = dict(YARN_BASE, rope_theta=theta, max_position_embeddings=max_pos,
                   rope_scaling=dict(scaling))
        cfg = LlamaConfig(**raw)
        cases.append(case(name, cfg, raw, None, "global", YARN_POSITIONS, None))

    import transformers
    doc = {
        "generated_by": "tools/oracle/rope_reference.py",
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "cases": cases,
    }
    with open(out, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    print(f"wrote {out}: {len(cases)} cases, {os.path.getsize(out)} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
