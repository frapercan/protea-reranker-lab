"""Apples-to-apples VALIDATION de-risk: score BOTH the original S2 trio (the
b21b187c champion, fit on v5 + overlays) and the new S2-assocfix trio (fit on
the corrected 46be427a export) on the SAME new c99db18 eval (220->227) split,
optimistic cafaeval recipe. Only the SAME-frame delta is meaningful.

If new >= orig on this common frame, the retrain is healthy -> proceed to TEST.
If new < orig, the retrain regressed even before TEST -> a caution flag.
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
OUT = "/home/frapercan/Thesis2/storage/fullgo_models/native_boosters_s2_assocfix/compare_validation"
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

NEW = "/home/frapercan/Thesis2/storage/fullgo_models/native_boosters_s2_assocfix"
ORIG = {
    "nk": "/home/frapercan/Thesis2/storage/fullgo_models/native_boosters_nklk_s2/ensemble_gbm_NK.txt",
    "lk": "/home/frapercan/Thesis2/storage/fullgo_models/native_boosters_nklk_s2/ensemble_gbm_LK.txt",
    "pk": "/home/frapercan/Thesis2/storage/fullgo_models/native_boosters_pk_s2/ensemble_gbm_PK.txt",
}


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


def score_trio(boosters, cols_cache, ia):
    pred = {c: {} for c in CATS}
    gt = {c: set() for c in CATS}
    for cat, prot, term, lab, Xfull in cols_cache:
        s = boosters[cat].predict(Xfull, num_threads=6)
        pc, gc = pred[cat], gt[cat]
        for k in range(len(prot)):
            key = (prot[k], term[k])
            v = s[k]
            if key not in pc or v > pc[key]:
                pc[key] = v
            if lab[k] == 1:
                gc.add(key)
    res = {}
    for c in CATS:
        pdir = Path(OUT) / f"_pd_{c}"
        pdir.mkdir(exist_ok=True)
        ptsv = f"{OUT}/pred_{c}.tsv"
        with open(ptsv, "w") as w:
            for (p, t), v in pred[c].items():
                w.write(f"{p}\t{t}\t{v:.6f}\n")
        gtsv = f"{OUT}/gt_{c}.tsv"
        with open(gtsv, "w") as w:
            for (p, t) in sorted(gt[c]):
                w.write(f"{p}\t{t}\n")
        link = pdir / "m.tsv"
        if link.exists():
            link.unlink()
        link.symlink_to(ptsv)
        df, _ = cafa_eval(OBO, str(pdir), gtsv, ia=IA_TSV, prop="fill", norm="cafa",
                          no_orphans=True, max_terms=None, th_step=0.01, n_cpu=1)
        sub = df.reset_index()
        col = "f_micro_w" if "f_micro_w" in sub.columns else "f_w"
        res[c] = float(sub.groupby("ns")[col].max().mean())
    return res


def main():
    ia = load_ia()
    new_b = {c: lgb.Booster(model_file=f"{NEW}/ensemble_gbm_{c.upper()}.txt") for c in CATS}
    orig_b = {c: lgb.Booster(model_file=ORIG[c]) for c in CATS}

    raw = CLIENT.get_object(BUCKET, f"{DATASET}/eval.parquet").read()
    pf = pq.ParquetFile(io.BytesIO(raw))
    cols = ["protein_accession", "go_term_id", "label", "category"] + LEAN
    # cache per-category (prot, term, label, X) once; both trios reuse it
    cache = []
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
            cache.append((c, [prot[i] for i in idx.tolist()], [term[i] for i in idx.tolist()],
                          lab[idx], X))
        n += b.num_rows
        print(f"  cached {n} rows", flush=True)

    print("scoring NEW (assocfix) trio", flush=True)
    new_res = score_trio(new_b, cache, ia)
    print("scoring ORIG (champion) trio", flush=True)
    orig_res = score_trio(orig_b, cache, ia)

    new_mean = sum(new_res.values()) / 3
    orig_mean = sum(orig_res.values()) / 3
    out = {
        "frame": "validation 220->227 on c99db18-assoc (optimistic no-TOI), SAME frame both trios",
        "new_assocfix": {c: round(new_res[c], 6) for c in CATS} | {"mean": round(new_mean, 6)},
        "orig_champion": {c: round(orig_res[c], 6) for c in CATS} | {"mean": round(orig_mean, 6)},
        "delta_mean": round(new_mean - orig_mean, 6),
        "delta_pk": round(new_res["pk"] - orig_res["pk"], 6),
    }
    json.dump(out, open(f"{OUT}/compare_result.json", "w"), indent=2)
    print(json.dumps(out, indent=2), flush=True)
    print(f"VERDICT validation: new MEAN={new_mean:.4f} vs orig MEAN={orig_mean:.4f} "
          f"delta={new_mean-orig_mean:+.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
