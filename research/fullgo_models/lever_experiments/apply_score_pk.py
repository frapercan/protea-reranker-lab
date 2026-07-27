"""Apply a lever PK booster to the validation-frame eval.parquet PK rows and score
with cafaeval, using the EXACT baseline recipe (so the number is comparable to the
0.4142 baseline). PK only.

Recipe (matches native_boosters_v5/validation_score):
  - features: the 69 numeric cols in v5 summary.json order
  - collapse duplicate (protein,term) candidate rows to ONE score by MAX
  - GT = label==1 PK rows (cafaeval propagates, prop=fill)
  - cafa_eval(OBO, pred_dir, gt, ia=IA, prop=fill, norm=cafa, no_orphans=True,
              max_terms=None, th_step=0.01, n_cpu=1)  [no TOI, no PK-exclude]
  - f_micro_w PK = mean over the 3 namespaces of the per-ns MAX f_micro_w
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq
from cafaeval.evaluation import cafa_eval
from minio import Minio

HERE = "/home/frapercan/Thesis2/storage/fullgo_models"
V5 = f"{HERE}/native_boosters_v5"
BUCKET = "protea"
EVAL_KEY = "datasets/fullgo-union-SELECT-160-220-227-v5/eval.parquet"
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"

with open(f"{V5}/summary.json") as fh:
    FEATURES = json.load(fh)["features"]
CLIENT = Minio("localhost:9000", access_key="minioadmin",
               secret_key="minioadmin", secure=False)


def apply_pk(booster_path: str, out_dir: str):
    booster = lgb.Booster(model_file=booster_path)
    assert booster.num_feature() == 69, booster.num_feature()
    raw = CLIENT.get_object(BUCKET, EVAL_KEY).read()
    pf = pq.ParquetFile(io.BytesIO(raw))
    cols = ["protein_accession", "go_term_id", "label", "category"] + FEATURES
    pred: dict = {}
    gt: set = set()
    n = 0
    for batch in pf.iter_batches(batch_size=400_000, columns=cols):
        d = batch.to_pydict()
        cat = np.asarray([str(x) for x in d["category"]])
        m = cat == "pk"
        if not m.any():
            n += len(cat)
            continue
        idx = np.nonzero(m)[0]
        prot = d["protein_accession"]
        term = d["go_term_id"]
        lab = np.asarray(d["label"], dtype=np.int8)
        X = np.empty((len(idx), len(FEATURES)), dtype=np.float32)
        for j, f in enumerate(FEATURES):
            col = d[f]
            arr = np.asarray(col)
            if arr.dtype == object or arr.dtype == bool:
                full = np.asarray([float(v) if v is not None else np.nan for v in col],
                                  dtype=np.float32)
            else:
                full = arr.astype(np.float32)
            X[:, j] = full[idx]
        scores = booster.predict(X, num_threads=2)
        for k, s in zip(idx.tolist(), scores.tolist()):
            key = (prot[k], term[k])
            if key not in pred or s > pred[key]:
                pred[key] = s
            if lab[k] == 1:
                gt.add(key)
        n += len(cat)
        del d, X
        print(f"    ...{n} rows", flush=True)
    pp = Path(out_dir) / "pred_pk.tsv"
    gg = Path(out_dir) / "gt_pk.tsv"
    with open(pp, "w") as w:
        for (p, t), s in pred.items():
            w.write(f"{p}\t{t}\t{s:.6f}\n")
    with open(gg, "w") as w:
        for (p, t) in sorted(gt):
            w.write(f"{p}\t{t}\n")
    print(f"pred_pairs={len(pred)} gt_pairs={len(gt)}", flush=True)
    return str(pp), str(gg)


def score_pk(out_dir: str) -> tuple[float, dict]:
    # absolute paths: cafaeval resolves the prediction file from its own CWD, so a
    # relative symlink target breaks. Resolve everything to absolute.
    base = Path(out_dir).resolve()
    pred = base / "pred_pk.tsv"
    gt = base / "gt_pk.tsv"
    pdir = base / "_pred_dir_pk"
    pdir.mkdir(exist_ok=True)
    link = pdir / "m.tsv"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(pred)
    df, _ = cafa_eval(OBO, str(pdir), str(gt), ia=IA, prop="fill", norm="cafa",
                      no_orphans=True, max_terms=None, th_step=0.01, n_cpu=1)
    sub = df.reset_index()
    col = "f_micro_w" if "f_micro_w" in sub.columns else "f_w"
    per_ns = sub.groupby("ns")[col].max()
    return float(per_ns.mean()), {ns: float(v) for ns, v in per_ns.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--booster", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    apply_pk(args.booster, args.out)
    v, per_ns = score_pk(args.out)
    print(f"PK f_micro_w = {v:.4f}  per-ns={ {k: round(x,4) for k,x in per_ns.items()} }",
          flush=True)
    out = {"tag": args.tag, "pk_f_micro_w": round(v, 6),
           "per_ns": {k: round(x, 6) for k, x in per_ns.items()},
           "baseline": 0.4142, "delta": round(v - 0.4142, 6)}
    with open(Path(args.out) / f"result_{args.tag}.json", "w") as w:
        json.dump(out, w, indent=2)
    print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
