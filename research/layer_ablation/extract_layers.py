"""Per-layer ankh-base embedding extractor for the representation ablation.

Runs ankh-base with output_hidden_states over the substrate sequences, mean-pools
(mask-aware) each of ~6 layers sampled across depth, saves one matrix per layer.
bfloat16 on CUDA (fp16 LayerNorm -> NaN). Cap 2048 (matches cached-embedding limit).
"""
import json, os, sys, numpy as np, torch
from transformers import AutoTokenizer, T5EncoderModel

W = "/home/frapercan/Thesis2/storage/layer_ablation"
MODEL = "ElnaggarLab/ankh-base"
MAXLEN = 2048
BATCH = 8
OUT = os.path.join(W, "emb_ankh_base"); os.makedirs(OUT, exist_ok=True)

seqs = json.load(open(os.path.join(W, "seqs.json")))
accs = sorted(seqs)
print(f"proteins: {len(accs)}", flush=True)

dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
model = T5EncoderModel.from_pretrained(MODEL, output_hidden_states=True,
                                       torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32)
model.eval().to(dev)

# probe layer count on one seq
with torch.no_grad():
    t = tok(["M A K"], return_tensors="pt", padding=True).to(dev)
    nlayers = len(model(**t).hidden_states)  # embeddings + each encoder layer
sampled = sorted(set(int(round(f * (nlayers - 1))) for f in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)))
print(f"hidden_states: {nlayers} | sampled layers: {sampled}", flush=True)

banks = {L: np.zeros((len(accs), model.config.d_model), np.float32) for L in sampled}
for i in range(0, len(accs), BATCH):
    chunk = accs[i:i + BATCH]
    batch_seqs = [" ".join(seqs[a][:MAXLEN]) for a in chunk]  # ankh: space-separated residues
    enc = tok(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN + 2).to(dev)
    with torch.no_grad():
        hs = model(**enc).hidden_states
    mask = enc["attention_mask"].unsqueeze(-1).to(torch.float32)  # (B,L,1)
    denom = mask.sum(1).clamp(min=1)
    for L in sampled:
        h = hs[L].to(torch.float32)
        pooled = (h * mask).sum(1) / denom  # mask-aware mean
        banks[L][i:i + len(chunk)] = pooled.cpu().numpy()
    if (i // BATCH) % 50 == 0:
        print(f"  {i}/{len(accs)}", flush=True)

for L in sampled:
    # float32 storage: middle-layer "massive activations" reach ~65k and overflow
    # float16 (max 65504) to Inf. Keep full precision.
    np.save(os.path.join(OUT, f"layer_{L}.npy"), banks[L].astype(np.float32))
json.dump({"accs": accs, "sampled_layers": sampled, "nlayers": nlayers, "dim": model.config.d_model},
          open(os.path.join(OUT, "meta.json"), "w"))
print(f"DONE: saved {len(sampled)} layers x {len(accs)} proteins -> {OUT}", flush=True)
