#!/usr/bin/env python3
"""Write the FP8 quantizer fixtures that `reng-fp8` tests against.

The oracle is PyTorch's own `float8_e4m3fn` / `float8_e5m2` cast, which
rounds to nearest even at the same exponent biases (7 and 15) as the Gaudi2
cast kernels. The one difference is the top exponent field: Gaudi reserves
it for infinity and NaN, so its largest finite E4M3 magnitude is 240 while
`float8_e4m3fn` runs to 448, and Gaudi's clipping cast saturates where
torch overflows. The fixture therefore stores, per case, the torch bytes
with every reserved-exponent byte replaced by the saturated one, and counts
how many bytes that touched.

Rows of a real checkpoint go in too (read straight out of the safetensors
file with numpy, no torch and no transformers), with the per-row mean and
maximum relative error of the round trip, so the Rust side checks its error
statistics against numpy's.

    python3 tools/fp8_fixtures.py --model <hf_model_dir> \
        --out crates/reng-fp8/fixtures/fp8_cast.json
"""

import argparse
import json
import os
import struct
import sys

import numpy as np
import torch

E4M3 = ("e4m3", torch.float8_e4m3fn, 240.0, 0x78, 0x77)
E5M2 = ("e5m2", torch.float8_e5m2, 57344.0, 0x7C, 0x7B)


def bf16_bits(x):
    """Round a float32 array to bf16, returning the bit patterns (RNE)."""
    b = x.astype(np.float32).view(np.uint32)
    bias = 0x7FFF + ((b >> 16) & 1)
    return ((b + bias) >> 16).astype(np.uint16)


def bf16_to_f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def cast(values, fmt):
    """Torch's fp8 bytes for `values`, and the same with Gaudi saturation."""
    _, dtype, maxf, reserved, sat = fmt
    t = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))
    raw = t.to(dtype).view(torch.uint8).numpy().copy()
    # A byte whose exponent field is all ones is out of Gaudi's finite
    # range; the clipping cast returns the largest finite magnitude.
    over = (raw & reserved) == reserved
    clipped = np.where(over, (raw & 0x80) | sat, raw).astype(np.uint8)
    return raw, clipped, int(over.sum())


def case(name, fmt, w_f32, scaling, share=None):
    """One fixture case: bf16 weights, per-row scales, expected bytes."""
    name_str, _, maxf, _, _ = fmt
    bits = bf16_bits(w_f32)
    w = bf16_to_f32(bits)
    rows, cols = w.shape
    default_bias = 7 if name_str == "e4m3" else 15
    exp_bias = default_bias
    if scaling == "pcs":
        absmax = np.abs(w).max(axis=1)
        scales = np.where(absmax > 0, absmax / np.float32(maxf), np.float32(1.0))
        scales = scales.astype(np.float32)
    elif scaling == "unit":
        scales = np.ones(rows, dtype=np.float32)
        exp_bias = default_bias
    elif scaling == "hw":
        # One power-of-16 factor for the whole matrix, expressed as the
        # exponent bias the MME descriptor takes: the largest of
        # {15, 11, 7, 3} whose range still covers the tensor absmax.
        absmax = float(np.abs(w).max())
        exp_bias = default_bias
        for b in (15, 11, 7, 3):
            if absmax <= maxf * 2.0 ** (default_bias - b):
                exp_bias = b
                break
        else:
            exp_bias = 3
        scales = np.full(
            rows, np.float32(2.0 ** (default_bias - exp_bias)), dtype=np.float32
        )
    else:
        raise SystemExit(f"unknown scaling {scaling}")
    scaled = (w / scales[:, None]).astype(np.float32)
    raw, clipped, over = cast(scaled, fmt)
    # The round trip, in numpy, for the error statistics.
    back = (
        torch.from_numpy(clipped.copy())
        .view(fmt[1])
        .to(torch.float32)
        .numpy()
        .reshape(rows, cols)
        * scales[:, None]
    )
    mean_rel, max_rel = [], []
    for r in range(rows):
        nz = w[r] != 0
        if not nz.any():
            mean_rel.append(0.0)
            max_rel.append(0.0)
            continue
        rel = np.abs((back[r][nz] - w[r][nz]) / w[r][nz])
        # float64 accumulation, so the Rust side's f64 sum agrees.
        mean_rel.append(float(rel.astype(np.float64).mean()))
        max_rel.append(float(rel.max()))
    out = {
        "name": name,
        "format": name_str,
        "scaling": scaling,
        "exp_bias": exp_bias,
        "rows": rows,
        "cols": cols,
        "bf16_hex": "" if share else "".join(f"{v:04x}" for v in bits.reshape(-1)),
        "bf16_from": share or "",
        "scale_bits": [int(v) for v in scales.view(np.uint32)],
        "codes_hex": "".join(f"{v:02x}" for v in clipped.reshape(-1)),
        "torch_bytes_saturated": over,
        "mean_rel": mean_rel,
        "max_rel": max_rel,
    }
    if over:
        # The bytes torch itself produced, kept so the Rust test can show
        # exactly which of them Gaudi's clipping cast saturates.
        out["torch_raw_hex"] = "".join(f"{v:02x}" for v in raw.reshape(-1))
    return out


def read_safetensors(path, names):
    """Named bf16 tensors of a safetensors file, as (bits, shape) pairs."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for name in names:
            meta = header[name]
            if meta["dtype"] != "BF16":
                raise SystemExit(f"{name}: dtype {meta['dtype']}, expected BF16")
            start, end = meta["data_offsets"]
            f.seek(base + start)
            buf = f.read(end - start)
            bits = np.frombuffer(buf, dtype="<u2").reshape(meta["shape"])
            out[name] = bits
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="HF model directory with model.safetensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=4, help="checkpoint rows per tensor")
    args = ap.parse_args()

    rng = np.random.default_rng(20260906)
    cases = []

    # Uniform weights, the plain case.
    cases.append(case("random_uniform", E4M3, rng.uniform(-1, 1, (8, 128)), "pcs"))
    # Llama-like: a narrow normal with a few outlier channels, which is
    # what per-output-channel scaling is for.
    w = rng.normal(0, 0.02, (8, 128))
    w[3] *= 40.0
    w[5, ::17] *= 25.0
    cases.append(case("llama_like", E4M3, w, "pcs"))
    # Rows spanning six decades: the scales must follow each row.
    decades = np.array([1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2])
    cases.append(
        case("wide_range", E4M3, rng.uniform(-1, 1, (6, 64)) * decades[:, None], "pcs")
    )
    # Unit scaling with magnitudes past 240: the saturation case, where
    # torch's own bytes and Gaudi's differ.
    sat = rng.uniform(-1, 1, (4, 32)) * 700.0
    cases.append(case("saturation_unit", E4M3, sat, "unit"))
    # E5M2, both schemes.
    cases.append(case("e5m2_random", E5M2, rng.uniform(-1, 1, (8, 128)), "pcs"))
    cases.append(
        case("e5m2_saturation_unit", E5M2, rng.uniform(-1, 1, (4, 32)) * 2e5, "unit")
    )

    if args.model:
        path = f"{args.model}/model.safetensors"
        names = [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.15.mlp.gate_proj.weight",
            "model.layers.29.mlp.down_proj.weight",
        ]
        tensors = read_safetensors(path, names)
        for name in names:
            bits = tensors[name][: args.rows]
            short = name.replace("model.layers.", "l").replace(".weight", "")
            short = short.replace("self_attn.", "").replace("mlp.", "")
            cases.append(case(f"ckpt_{short}", E4M3, bf16_to_f32(bits), "pcs"))
            cases.append(
                case(
                    f"ckpt_{short}_hw",
                    E4M3,
                    bf16_to_f32(bits),
                    "hw",
                    share=f"ckpt_{short}",
                )
            )

    doc = {
        "generator": "tools/fp8_fixtures.py",
        "torch": torch.__version__,
        "numpy": np.__version__,
        # The model's own name only: nothing in the repo names a path on
        # the machine that generated the fixture.
        "model": os.path.basename(args.model.rstrip("/")) if args.model else "",
        "cases": cases,
    }
    with open(args.out, "w") as f:
        json.dump(doc, f)
        f.write("\n")
    total = sum(c["rows"] * c["cols"] for c in cases)
    print(f"{args.out}: {len(cases)} cases, {total} values, torch {torch.__version__}")
    for c in cases:
        print(
            f"  {c['name']:<28} {c['format']} {c['scaling']:<4} "
            f"{c['rows']}x{c['cols']} bias={c['exp_bias']} "
            f"saturated={c['torch_bytes_saturated']} "
            f"mean_rel={np.mean(c['mean_rel']):.5f} max_rel={max(c['max_rel']):.5f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
