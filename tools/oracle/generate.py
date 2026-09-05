"""HF greedy-generation reference (CPU, f32).

usage: generate.py <model_dir> <out.json> <n_new> <token_id> [token_id ...]
Writes the prompt, the greedily generated ids, the decoded text, and for each
generated step the f32 top-8 candidates with their logits (teacher-forced
over the reference sequence), so a bf16 engine can tell a near-tie from a
real divergence.
"""
import json, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
d, out, n_new = sys.argv[1], sys.argv[2], int(sys.argv[3])
ids = [int(t) for t in sys.argv[4:]]
tok = AutoTokenizer.from_pretrained(d)
m = AutoModelForCausalLM.from_pretrained(d, torch_dtype=torch.float32)
m.eval()
with torch.no_grad():
    g = m.generate(torch.tensor([ids]), max_new_tokens=n_new, do_sample=False,
                   pad_token_id=tok.eos_token_id)
    full = g[0].tolist()
    lg = m(torch.tensor([full])).logits[0]  # [T, vocab], teacher forced
new = full[len(ids):]
steps = []
for i, t in enumerate(new):
    row = lg[len(ids) - 1 + i]
    v, ix = torch.topk(row, 8)
    steps.append({"top1": ix[0].item(), "top2": ix[1].item(),
                  "margin": (v[0] - v[1]).item(),
                  "top_ids": ix.tolist(), "top_logits": v.tolist()})
    assert ix[0].item() == t, (i, ix[0].item(), t)
res = {"prompt": ids, "generated": new, "steps": steps,
       "text": tok.decode(full), "prompt_text": tok.decode(ids),
       "new_text": tok.decode(new)}
json.dump(res, open(out, "w"))
print("prompt:", repr(res["prompt_text"]))
print("generated ids:", new)
print("margins:", [round(s["margin"], 3) for s in steps])
print("new text:", repr(res["new_text"]))
