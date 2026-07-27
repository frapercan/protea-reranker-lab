"""Extract ProTrek's text-aligned protein SEQUENCE embedding for a sequence set.

ProTrek (westlake-repl/ProTrek) is a trimodal (sequence/structure/text) protein
model whose protein-SEQUENCE encoder (ESM2-650M + linear head) is contrastively
aligned to a text encoder. `model.get_protein_repr([aa_seq])` returns the
L2-normalized, text-aligned sequence embedding (1024-d) -- SEQUENCE ONLY, no
structure input required. This is the ProTrek analogue of ProtST's protein_feature,
used as a 2nd independent text-aligned datapoint for a GO-transfer study.

Usage:
  extract_protrek.py SEQS_JSON OUT_PREFIX

SEQS_JSON: {accession: aa_sequence, ...}
Saves OUT_PREFIX_protrek.npy (float32, N x 1024) + OUT_PREFIX_meta.json {"accs":[...],"dim":D}

Env overrides:
  PROTREK_REPO     path to the cloned ProTrek repo (default: alongside this file's tools dir)
  PROTREK_WEIGHTS  path to the ProTrek_650M weights dir (default: <repo>/weights/ProTrek_650M)
"""
import sys, os, json, time, numpy as np, torch

REPO = os.environ.get("PROTREK_REPO", "/home/frapercan/Thesis2/storage/tools/ProTrek")
WEIGHTS = os.environ.get("PROTREK_WEIGHTS", os.path.join(REPO, "weights", "ProTrek_650M"))
sys.path.insert(0, REPO)
from model.ProTrek.protrek_trimodal_model import ProTrekTrimodalModel

SEQS = sys.argv[1]; PREFIX = sys.argv[2]
MAXLEN = 1022        # ESM2 supports 1024 positions incl. <cls>/<eos>; cap residues at 1022
BATCH = 8

seqs = json.load(open(SEQS)); accs = sorted(seqs)
print(f"proteins: {len(accs)}", flush=True)

config = {
    "protein_config": os.path.join(WEIGHTS, "esm2_t33_650M_UR50D"),
    "text_config": os.path.join(WEIGHTS, "BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"),
    "structure_config": os.path.join(WEIGHTS, "foldseek_t30_150M"),
    "from_checkpoint": os.path.join(WEIGHTS, "ProTrek_650M.pt"),
}
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ProTrekTrimodalModel(**config).eval().to(device)

D = model.repr_dim  # 1024
out = np.zeros((len(accs), D), np.float32)
t0 = time.time()
with torch.no_grad():
    for i in range(0, len(accs), BATCH):
        chunk = accs[i:i + BATCH]
        batch_seqs = [seqs[a][:MAXLEN] for a in chunk]
        repr_ = model.get_protein_repr(batch_seqs, batch_size=BATCH, verbose=False)  # [b, D], L2-normed
        out[i:i + len(chunk)] = repr_.float().cpu().numpy()
        if (i // BATCH) % 80 == 0:
            print(f"  {i}/{len(accs)}  ({time.time() - t0:.1f}s)", flush=True)

np.save(PREFIX + "_protrek.npy", out)
json.dump({"accs": accs, "dim": int(D)}, open(PREFIX + "_meta.json", "w"))
print(f"DONE saved {PREFIX}_protrek.npy  shape={out.shape}  ({time.time() - t0:.1f}s)", flush=True)
