"""LAFA submission entrypoint for the fullgo KNN + classifier ensemble.

Contract (An Phan / CAFA_forever): read query FASTA, emit a 3-column TSV
``Query_ID  GO_Term  Score``. The container ships pre-trained weights and a
frozen t0 reference bundle (bind-mounted); it does NOT train at runtime.

Pipeline per query protein:
  1. mean-pooled embeddings from the 6 frozen PLMs -> 8320-d concat
  2. full-label classifier (classifier_6plm_asl.pt) -> per-term scores
  3. numpy KNN over the bundled reference pool -> composite score + vote
  4. per-category ensemble GBM over the DEPLOYABLE feature set -> final score

DEPLOYABLE FEATURE SET (container-computable): [knn_score, vote, clf_score,
knn_present, clf_present, IA, log_freq]. The lab's 0.349 ensemble additionally
used alignment (identity_nw/sw) and taxonomic_distance sub-features; those need
PROTEA method-runtime (NW/SW + taxonomy) and are DROPPED here. Retrain the GBMs
on this 7-feature set (scripts/retrain_deployable_gbm.py) before shipping; the
score change is minor (those features were low-importance) but train/serve must
match. See BUILD.md.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb

MODELS = Path("/app/models")           # baked into the image
BUNDLE = Path("/app/data")             # bind-mounted frozen t0 reference
# config.yaml plm order = the concat order used at training time
PLM_HF = [
    "ElnaggarLab/ankh-base", "facebook/esm2_t36_3B_UR50D", "ElnaggarLab/ankh-large",
    "facebook/esm2_t33_650M_UR50D", "esmc_600m", "Rostlab/prot_t5_xl_half_uniref50-enc",
]


class MLP(nn.Module):
    def __init__(self, di, h, o):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(di, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.2), nn.Linear(h, o))

    def forward(self, x):
        return self.net(x)


def embed_queries(fasta, device):
    """Return {accession: 8320-d np.float32} mean-pooled 6-PLM concat.

    Each backend is loaded, run, and freed in turn to bound VRAM. Implemented
    against transformers / the esm package; see protea-backends for the exact
    pooling used at training time (mean over residues, no special tokens)."""
    raise NotImplementedError(
        "Wire to protea-backends embedders (mean pooling, same configs as config.yaml). "
        "Load each PLM in PLM_HF, mean-pool residue embeddings, hstack in order.")


def knn_features(query_emb, ref):
    """numpy KNN (K=30) over the reference Ankh-base block -> per (query, term)
    composite score + vote fraction, matching the platform's composite formula
    on the deployable feature subset. ref = bundled {emb, accs, labels}."""
    raise NotImplementedError(
        "Cosine KNN over ref embeddings; transfer ref labels; composite = weighted "
        "mean of [embedding_similarity, neighbor_vote_fraction]; vote = fraction of "
        "the K neighbours carrying the term. See fullgo/REPRODUCE.md step 1.")


def main():
    ap = argparse.ArgumentParser(description="fullgo ensemble LAFA predictor")
    ap.add_argument("--query_file", "-q", required=True)
    ap.add_argument("--train_sequences", default=None)
    ap.add_argument("--annot_file", "-a", default=None)
    ap.add_argument("--graph", default=None)
    ap.add_argument("--output_file", "-o", required=True)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(MODELS / "classifier_6plm_asl.pt", map_location=dev)
    vocab = ckpt["vocab"]; mu = ckpt["mu"]; sd = ckpt["sd"]
    clf = MLP(ckpt["in_dim"], ckpt["hidden"], len(vocab)).to(dev)
    clf.load_state_dict(ckpt["state_dict"]); clf.eval()
    gbm = {c: lgb.Booster(model_file=str(MODELS / f"ensemble_gbm_{c}.txt")) for c in ("NK", "LK", "PK")}
    ia = {ln.split("\t")[0]: float(ln.split("\t")[1]) for ln in open(BUNDLE / "IA.tsv") if "\t" in ln}
    freq = {ln.split("\t")[0]: int(ln.split("\t")[1]) for ln in open(MODELS / "v227_exp_freq.tsv") if "\t" in ln}

    emb = embed_queries(args.query_file, dev)                  # {acc: 8320-d}
    ref = np.load(BUNDLE / "reference_pool.npz", allow_pickle=True)

    with open(args.output_file, "w") as out:
        for acc, e in emb.items():
            x = ((torch.tensor(e) - mu) / sd).to(dev)
            with torch.no_grad():
                clf_scores = torch.sigmoid(clf(x.unsqueeze(0)))[0].cpu().numpy()
            clf_top = {vocab[j]: float(clf_scores[j]) for j in np.argsort(-clf_scores)[:100]}
            knn_top = knn_features(e, ref)                      # {term: (knn_score, vote)}
            cand = set(clf_top) | set(knn_top)
            # category is assigned by the maintainer's frame; emit one score per cand.
            feats, terms = [], []
            for t in cand:
                ks, vote = knn_top.get(t, (0.0, 0.0)); cs = clf_top.get(t, 0.0)
                feats.append([ks, vote, cs, 1.0 if t in knn_top else 0.0,
                              1.0 if t in clf_top else 0.0, ia.get(t, 0.0), math.log1p(freq.get(t, 0))])
                terms.append(t)
            # NK booster as the default arm (see BUILD.md on per-category routing)
            pred = gbm["NK"].predict(np.array(feats, np.float32))
            for t, s in zip(terms, pred):
                if s >= 0.01:
                    out.write(f"{acc}\t{t}\t{s:.6f}\n")


if __name__ == "__main__":
    main()
