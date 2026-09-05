# Oracle

CPU f32 references from Hugging Face transformers, used to validate the
engine. They never run inside the engine; they produce JSON that
`reng-prefill --ref` and `reng-generate --ref` compare against.

## Setup

```console
python3 -m venv oracle-venv
oracle-venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
oracle-venv/bin/pip install transformers safetensors
```

Tested with torch 2.14.0+cpu and transformers 5.16.1.

## Use

```console
# per-position argmax and last-position logits for a prompt
python oracle.py <model_dir> ref.json 504 3575 282 4649 314
reng-prefill <model_dir> engine.json --ref ref.json 504 3575 282 4649 314

# greedy generation with per-step top-1/top-2 margins
python generate.py <model_dir> gen.json 8 504 3575 282 4649 314
reng-generate <model_dir> engine.json 8 --ref gen.json 504 3575 282 4649 314
```

`reng-generate --ref` is teacher-forced: it scores the reference prefix at
every step, so one near-tie does not desynchronise the rest of the check. A
mismatch fails unless the engine's token is within `--margin` logits
(default 0.5) of the reference's best candidate in the reference's f32
logits; such candidates are within bf16 rounding of each other and the
mismatch is reported as a near-tie.
