"""Score the v6-ON azucar boosters on the 220->227 validation eval split with the
SAME no-TOI cafaeval recipe the lean experiment used, to re-validate lean-vs-v6 on
the validation frame (methodology cleanup; the lean-vs-v6 call was made on TEST).
Lean control = 0.5328 mean (from selfprior_ia_experiment)."""
import io, json, os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import lightgbm as lgb
from minio import Minio
from cafaeval.evaluation import cafa_eval

BUCKET = "protea"
BASE = "datasets/fullgo-union-SELECT-160-220-227-v5"
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
CLIENT = Minio("localhost:9000", access_key="minioadmin", secret_key="minioadmin", secure=False)
CATS = ("nk", "lk", "pk")
BDIR = "/home/frapercan/Thesis2/storage/fullgo_models/native_boosters_azucar"
OUT = f"{BDIR}/validation_score_220_227"
os.makedirs(OUT, exist_ok=True)

boosters = {c: lgb.Booster(model_file=f"{BDIR}/ensemble_gbm_{c.upper()}.txt") for c in CATS}
feats = {c: boosters[c].feature_name() for c in CATS}
allfeats = sorted({f for c in CATS for f in feats[c]})
print(f"azucar features per cat: {len(feats['pk'])}", flush=True)

raw = CLIENT.get_object(BUCKET, f"{BASE}/eval.parquet").read()
pf = pq.ParquetFile(io.BytesIO(raw))
cols = ["protein_accession", "go_term_id", "label", "category"] + allfeats
pred = {c: {} for c in CATS}
gt = {c: set() for c in CATS}


def tofloat(col):
    return np.array([float(x) if (x is not None and x != "") else np.nan for x in col], dtype=np.float32)


n = 0
for b in pf.iter_batches(batch_size=1_000_000, columns=cols):
    d = b.to_pydict()
    cat = np.array([str(x) for x in d["category"]])
    prot, term = d["protein_accession"], d["go_term_id"]
    lab = np.array(d["label"], dtype=np.int8)
    fcols = {}
    for f in allfeats:
        col = d[f]
        try:
            fcols[f] = np.asarray(col, dtype=np.float32)
        except (TypeError, ValueError):
            fcols[f] = tofloat(col)
    for c in CATS:
        m = cat == c
        if not m.any():
            continue
        idx = np.nonzero(m)[0]
        X = np.column_stack([fcols[f][idx] for f in feats[c]])
        scores = boosters[c].predict(X, num_threads=4)
        for k, s in zip(idx.tolist(), scores.tolist()):
            key = (prot[k], term[k])
            if key not in pred[c] or s > pred[c][key]:
                pred[c][key] = s
            if lab[k] == 1:
                gt[c].add(key)
    n += b.num_rows
    print(f"  scored {n} rows", flush=True)

res = {}
for c in CATS:
    with open(f"{OUT}/pred_{c}.tsv", "w") as w:
        for (p, t), s in pred[c].items():
            w.write(f"{p}\t{t}\t{s:.6f}\n")
    with open(f"{OUT}/gt_{c}.tsv", "w") as w:
        for (p, t) in sorted(gt[c]):
            w.write(f"{p}\t{t}\n")
    pdir = Path(OUT) / f"_pd_{c}"
    pdir.mkdir(exist_ok=True)
    link = pdir / "m.tsv"
    if link.exists():
        link.unlink()
    link.symlink_to(f"{OUT}/pred_{c}.tsv")
    df, _ = cafa_eval(OBO, str(pdir), f"{OUT}/gt_{c}.tsv", ia=IA, prop="fill",
                      norm="cafa", no_orphans=True, max_terms=None, th_step=0.01, n_cpu=1)
    sub = df.reset_index()
    col = "f_micro_w" if "f_micro_w" in sub.columns else "f_w"
    res[c] = float(sub.groupby("ns")[col].max().mean())
    print(f"azucar(v6on) {c.upper()} f_micro_w={res[c]:.4f}", flush=True)

mean = sum(res.values()) / len(res)
print(f"AZUCAR v6-ON MEAN={mean:.4f}  (vs LEAN control 0.5328)", flush=True)
json.dump({"variant": "azucar_v6on", "f_micro_w": {c: round(res[c], 6) for c in CATS},
           "mean": round(mean, 6)}, open(f"{OUT}/result.json", "w"), indent=2)
