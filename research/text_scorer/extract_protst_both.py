"""Extract BOTH ProtST outputs for a sequence set: the text-aligned protein_feature
AND the raw ESM-1b mean-pool (mask-aware mean of residue_feature) -- the same-base
control that isolates the text-alignment contribution. Usage:
  extract_protst_both.py SEQS_JSON OUT_PREFIX
Saves OUT_PREFIX_protst.npy (text, 512-d) and OUT_PREFIX_esm1braw.npy (raw ESM, 512-d).
"""
import sys, json, os, numpy as np, torch
from transformers import AutoModel, AutoTokenizer

SEQS = sys.argv[1]; PREFIX = sys.argv[2]
MAXLEN = 1022; BATCH = 12
seqs = json.load(open(SEQS)); accs = sorted(seqs)
print(f"proteins: {len(accs)}", flush=True)
tok = AutoTokenizer.from_pretrained("facebook/esm1b_t33_650M_UR50S")
model = AutoModel.from_pretrained("mila-intel/ProtST-esm1b", trust_remote_code=True).eval().to("cuda")
sp = tok.all_special_ids
txt = np.zeros((len(accs), 512), np.float32)
raw = np.zeros((len(accs), 512), np.float32)
for i in range(0, len(accs), BATCH):
    chunk = accs[i:i + BATCH]
    enc = tok([seqs[a][:MAXLEN] for a in chunk], return_tensors="pt", padding=True,
              truncation=True, max_length=MAXLEN + 2).to("cuda")
    with torch.no_grad():
        out = model.protein_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    feat = (out.protein_feature if hasattr(out, "protein_feature") else out["protein_feature"]).float()
    res = (out.residue_feature if hasattr(out, "residue_feature") else out["residue_feature"]).float()
    # raw ESM mean-pool over non-special, non-pad tokens
    ids = enc["input_ids"]
    keep = enc["attention_mask"].bool().clone()
    for s in sp:
        keep &= (ids != s)
    m = keep.unsqueeze(-1).float()
    rawmean = (res * m).sum(1) / m.sum(1).clamp(min=1)
    txt[i:i + len(chunk)] = feat.cpu().numpy()
    raw[i:i + len(chunk)] = rawmean.cpu().numpy()
    if (i // BATCH) % 80 == 0:
        print(f"  {i}/{len(accs)}", flush=True)
np.save(PREFIX + "_protst.npy", txt)
np.save(PREFIX + "_esm1braw.npy", raw)
json.dump({"accs": accs, "dim": 512}, open(PREFIX + "_meta.json", "w"))
print(f"DONE saved {PREFIX}_protst.npy + _esm1braw.npy", flush=True)
