"""Extract ankh-base layers 10, 19 and 48 for the 100,000-protein training pool.

Layer 48 is the served champion's base and the in-experiment control. Layer 10 is the best
single fixed layer the ablation found. Layer 19 is the one the learned mix mildly preferred
when it refused to specialise. Extracting all three in one forward pass costs the same as
extracting one, so the expensive part happens once.

float32 storage, never float16: the mid layers carry massive activations (layer 38 peaks at
|490,564| on the reference set, and float16 tops out at 65,504, overflowing silently to Inf).
Compute stays bfloat16 on CUDA because fp16 LayerNorm goes NaN.

Writes one matrix per layer plus a meta.json pinning the accession order, so every later
arm sees exactly the same substrate.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, T5EncoderModel

W = Path("/home/frapercan/Thesis2/storage/layer_ablation")
OUT = W / "scale_pool_emb"
MODEL = "ElnaggarLab/ankh-base"
MAXLEN = 2048
BATCH = 8
LAYERS = [10, 19, 48]

OUT.mkdir(parents=True, exist_ok=True)
seqs = json.load(open(W / "scale_pool_seqs.json"))
accs = json.load(open(W / "scale_pool_meta.json"))["accs"]
assert all(a in seqs for a in accs), "pool meta and sequences disagree"
print(f"pool: {len(accs):,} proteins | layers {LAYERS}", flush=True)

dev = "cuda" if torch.cuda.is_available() else "cpu"
assert dev == "cuda", "refusing to run this on CPU: it would take days"
tok = AutoTokenizer.from_pretrained(MODEL)
model = T5EncoderModel.from_pretrained(
    MODEL, output_hidden_states=True, torch_dtype=torch.bfloat16).eval().to(dev)

banks = {L: np.zeros((len(accs), model.config.d_model), np.float32) for L in LAYERS}
for i in range(0, len(accs), BATCH):
    chunk = accs[i:i + BATCH]
    batch = [" ".join(seqs[a][:MAXLEN]) for a in chunk]
    enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
              max_length=MAXLEN + 2).to(dev)
    with torch.no_grad():
        hs = model(**enc).hidden_states
    mask = enc["attention_mask"].unsqueeze(-1).to(torch.float32)
    denom = mask.sum(1).clamp(min=1)
    for L in LAYERS:
        banks[L][i:i + len(chunk)] = ((hs[L].to(torch.float32) * mask).sum(1) / denom).cpu().numpy()
    if (i // BATCH) % 200 == 0:
        print(f"  {i}/{len(accs)}", flush=True)

for L in LAYERS:
    a = banks[L]
    assert np.isfinite(a).all(), f"layer {L} has non-finite values"
    np.save(OUT / f"layer_{L}.npy", a.astype(np.float32))
    print(f"  saved layer_{L}.npy {a.shape} |max|={np.abs(a).max():,.0f}", flush=True)
json.dump({"accs": accs, "layers": LAYERS}, open(OUT / "meta.json", "w"))
print(f"DONE -> {OUT}", flush=True)
