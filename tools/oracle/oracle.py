"""HF transformers reference for a Llama-style model on CPU (f32).

usage: oracle.py <model_dir> <out.json> <token_id> [token_id ...]
Writes per-position argmax and the full last-position logits.
"""
import json, sys, torch
from transformers import AutoModelForCausalLM
d, out, ids = sys.argv[1], sys.argv[2], [int(t) for t in sys.argv[3:]]
m = AutoModelForCausalLM.from_pretrained(d, torch_dtype=torch.float32)
m.eval()
with torch.no_grad():
    lg = m(torch.tensor([ids])).logits[0]  # [T, vocab]
res = {"argmax": lg.argmax(-1).tolist(),
       "last_logits": lg[-1].tolist(),
       "last_top5": torch.topk(lg[-1], 5).indices.tolist()}
json.dump(res, open(out, "w"))
print("argmax:", res["argmax"])
print("last top5:", res["last_top5"])
