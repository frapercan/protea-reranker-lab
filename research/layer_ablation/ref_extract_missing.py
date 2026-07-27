"""Extract the reference-set layers the crowning experiment needs but that
`ref_extract.py` never produced: L0, L19, L29, L38 (L10 and L48 already exist).

With all six sampled layers on both sides, a head can be trained on a learned mix
over layers rather than on the last layer alone, which is the base `d8979601`
happens to sit on and which the ablation flags as the worst fixed one.

float32 storage, never float16: mid-layer activations reach |440,611| and overflow
silently to Inf. Compute stays bfloat16 on CUDA (fp16 LayerNorm goes NaN).
"""
import json, os, numpy as np, torch
from transformers import AutoTokenizer, T5EncoderModel

W = "/home/frapercan/Thesis2/storage/layer_ablation"
MODEL = "ElnaggarLab/ankh-base"; MAXLEN = 2048; BATCH = 6
LAYERS = [0, 19, 29, 38]
OUT = os.path.join(W, "ref_emb"); os.makedirs(OUT, exist_ok=True)

seqs = json.load(open(os.path.join(W, "ref_seqs.json")))
accs = sorted(seqs)
# meta.json already pins the accession order for L10/L48; reuse it so every layer aligns.
prev = json.load(open(os.path.join(OUT, "meta.json")))["accs"]
assert prev == accs, "reference accession order changed; the layer banks would misalign"
print(f"reference proteins: {len(accs)} | extracting layers {LAYERS}", flush=True)

dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
model = T5EncoderModel.from_pretrained(
    MODEL, output_hidden_states=True,
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
    a = banks[L]
    assert np.isfinite(a).all(), f"layer {L} has non-finite values (float16 overflow?)"
    np.save(os.path.join(OUT, f"layer_{L}.npy"), a.astype(np.float32))
    print(f"  saved layer_{L}.npy {a.shape}", flush=True)
print(f"DONE: reference layers {LAYERS} -> {OUT}", flush=True)
