"""Extract ankh-base L10 (winner) + L48 (baseline) for the reference subset.

Same as extract_layers.py but only the two layers we need for the board-faithful
kNN confirm, float32 storage (middle-layer massive activations overflow float16).
"""
import json, os, numpy as np, torch
from transformers import AutoTokenizer, T5EncoderModel

W = "/home/frapercan/Thesis2/storage/layer_ablation"
MODEL = "ElnaggarLab/ankh-base"; MAXLEN = 2048; BATCH = 6
LAYERS = [10, 48]
OUT = os.path.join(W, "ref_emb"); os.makedirs(OUT, exist_ok=True)

seqs = json.load(open(os.path.join(W, "ref_seqs.json")))
accs = sorted(seqs)
print(f"reference proteins: {len(accs)}", flush=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
model = T5EncoderModel.from_pretrained(MODEL, output_hidden_states=True,
                                       torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32).eval().to(dev)
banks = {L: np.zeros((len(accs), model.config.d_model), np.float32) for L in LAYERS}
for i in range(0, len(accs), BATCH):
    chunk = accs[i:i + BATCH]
    bs = [" ".join(seqs[a][:MAXLEN]) for a in chunk]
    enc = tok(bs, return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN + 2).to(dev)
    with torch.no_grad():
        hs = model(**enc).hidden_states
    mask = enc["attention_mask"].unsqueeze(-1).to(torch.float32); denom = mask.sum(1).clamp(min=1)
    for L in LAYERS:
        banks[L][i:i + len(chunk)] = ((hs[L].to(torch.float32) * mask).sum(1) / denom).cpu().numpy()
    if (i // BATCH) % 100 == 0:
        print(f"  {i}/{len(accs)}", flush=True)
for L in LAYERS:
    np.save(os.path.join(OUT, f"layer_{L}.npy"), banks[L].astype(np.float32))
json.dump({"accs": accs}, open(os.path.join(OUT, "meta.json"), "w"))
print(f"DONE: reference L{LAYERS} saved -> {OUT}", flush=True)
