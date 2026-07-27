"""De-risk on VALIDATION: score the freshly-trained S2-assocfix trio on the new
eval (220->227) split with the SAME optimistic cafaeval recipe the lean/both
experiment used (no-TOI, prop=fill, norm=cafa, no_orphans, MAX-collapse, IA).

Reference baselines on this frame (selfprior_ia_experiment/summary.json):
  control  MEAN 0.5328  (PK 0.3605)
  both     MEAN 0.5439  (PK 0.3720)   <- lean-31 + IA, pre scale_pos_weight
The S2 lever (scale_pos_weight) adds a small PK lift on top of `both`.

Only the SAME-recipe relative comparison is meaningful (optimistic magnitudes).
If the trio is broken (way below ~0.54 MEAN / PK far below 0.37), STOP before TEST.

Reads the new dataset eval.parquet natively (fixed self_prior + corrected
association already in the parquet) and overlays IA[go_term_id]. Logs the
validation MEAN to the same MLflow run.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq
from cafaeval.evaluation import cafa_eval
from minio import Minio

DATASET = "datasets/fullgo-union-SELECT-160-220-227-c99db18-assoc"
BUCKET = "protea"
IA_TSV = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
BDIR = "/home/frapercan/Thesis2/storage/fullgo_models/native_boosters_s2_assocfix"
OUT = f"{BDIR}/validation_score_220_227"
os.makedirs(OUT, exist_ok=True)
CLIENT = Minio("localhost:9000", access_key="minioadmin", secret_key="minioadmin", secure=False)
CATS = ("nk", "lk", "pk")

LEAN = [
    "distance", "identity_nw", "similarity_nw", "alignment_score_nw", "gaps_pct_nw",
    "alignment_length_nw", "identity_sw", "similarity_sw", "alignment_score_sw",
    "gaps_pct_sw", "alignment_length_sw", "length_query", "length_ref",
    "taxonomic_distance", "taxonomic_common_ancestors", "vote_count", "k_position",
    "go_term_frequency", "ref_annotation_density", "neighbor_distance_std",
    "neighbor_vote_fraction", "neighbor_min_distance", "neighbor_mean_distance",
    "knn_present", "classifier_score", "classifier_present", "self_prior_score",
    "association_total", "association_cross", "association_present",
]
NAMES = LEAN + ["IA"]


def load_ia():
    ia = {}
    for ln in open(IA_TSV):
        p = ln.rstrip("\n").split("\t")
        if len(p) >= 2:
            try:
                ia[p[0]] = float(p[1])
            except ValueError:
                pass
    return ia


def main():
    boosters = {c: lgb.Booster(model_file=f"{BDIR}/ensemble_gbm_{c.upper()}.txt") for c in CATS}
    for c in CATS:
        assert boosters[c].feature_name() == NAMES, f"{c} feature mismatch"
    ia = load_ia()

    raw = CLIENT.get_object(BUCKET, f"{DATASET}/eval.parquet").read()
    pf = pq.ParquetFile(io.BytesIO(raw))
    cols = ["protein_accession", "go_term_id", "label", "category"] + LEAN
    pred = {c: {} for c in CATS}
    gt = {c: set() for c in CATS}

    n = 0
    for b in pf.iter_batches(batch_size=2_000_000, columns=cols):
        cat = np.asarray(b.column("category").to_numpy(zero_copy_only=False), dtype=object)
        prot = b.column("protein_accession").to_pylist()
        term = b.column("go_term_id").to_pylist()
        lab = b.column("label").to_numpy(zero_copy_only=False).astype(np.int8)
        fcols = {f: np.asarray(b.column(f).to_numpy(zero_copy_only=False), dtype=np.float64).astype(np.float32)
                 for f in LEAN}
        iav = np.fromiter((ia.get(t, 0.0) for t in term), dtype=np.float32, count=len(term))
        for c in CATS:
            m = cat == c
            if not m.any():
                continue
            idx = np.nonzero(m)[0]
            X = np.empty((idx.size, len(NAMES)), dtype=np.float32)
            for j, f in enumerate(LEAN):
                X[:, j] = fcols[f][idx]
            X[:, len(LEAN)] = iav[idx]
            scores = boosters[c].predict(X, num_threads=6)
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
        df, _ = cafa_eval(OBO, str(pdir), f"{OUT}/gt_{c}.tsv", ia=IA_TSV, prop="fill",
                          norm="cafa", no_orphans=True, max_terms=None, th_step=0.01, n_cpu=1)
        sub = df.reset_index()
        col = "f_micro_w" if "f_micro_w" in sub.columns else "f_w"
        res[c] = float(sub.groupby("ns")[col].max().mean())
        print(f"s2_assocfix {c.upper()} f_micro_w={res[c]:.4f}", flush=True)

    mean = sum(res.values()) / len(res)
    out = {"variant": "s2_assocfix_validation", "frame": "validation 220->227 (optimistic no-TOI)",
           "f_micro_w": {c: round(res[c], 6) for c in CATS}, "mean": round(mean, 6),
           "reference_both": {"mean": 0.5439, "pk": 0.3720}}
    json.dump(out, open(f"{OUT}/result.json", "w"), indent=2)
    print(f"S2-ASSOCFIX VALIDATION MEAN={mean:.4f}  (ref both MEAN 0.5439 / PK 0.3720)", flush=True)

    # log to the training MLflow run
    try:
        import mlflow
        rid_path = f"{BDIR}/mlflow_run_id.txt"
        if os.path.exists(rid_path):
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
            with mlflow.start_run(run_id=open(rid_path).read().strip()):
                for c in CATS:
                    mlflow.log_metric(f"val_fmicrow_{c}", res[c])
                mlflow.log_metric("val_fmicrow_mean", mean)
                mlflow.log_artifact(f"{OUT}/result.json")
    except Exception as e:
        print(f"mlflow log skipped: {e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
