#!/usr/bin/env python
"""Track C1: per-category reranker on the clean A4 temporal frame.

Trains a per-category LightGBM LambdaMART combiner over the EvidenceScorer
score vector (ADR-D43, ``src/protea_reranker_lab/combiner.py``) on the fresh
``fullgo-clean-A4-train225-val`` export (train snapshot pairs up to v220-v225;
the current 73-feature live schema), then asks the single question this loop
exists for: does reranking LIFT IA-weighted ``f_micro_w`` over the raw-KNN
score, per category x aspect, on the validation frame v225 -> v227.

Pipeline
--------
1. TRAIN: one lambdarank booster per category (NK / LK / PK) over the
   per-category score vector. NK drops the priors per the combiner contract.
   The zero-filled ``association_* / classifier_score`` columns are dropped
   (compute flags were off in this export). Internal early-stopping holdout =
   the last train snapshot pair (v220-v225); the eval frame stays untouched.
   Per-iteration ndcg is streamed to MLflow (experiment
   ``reranker-clean-A4-train225``).
2. EVAL: cafaeval ``f_micro_w`` (via the PROTEA venv) for the raw-KNN
   baseline (``neighbor_vote_fraction``) vs the reranked score, per cell.
3. CI: a numpy reimplementation of the IA-weighted micro-F (propagation +
   no_orphans + max_terms, matching the cafaeval contract) validated against
   the cafaeval point estimate, then a per-protein paired bootstrap for 95%
   CIs on the f_micro_w delta. PK cells additionally report IA-weighted
   precision vs recall at the f_micro_w-optimal threshold.

Run with the lab venv python (lightgbm / pyarrow / mlflow); the cafaeval
subprocess uses the PROTEA venv.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from protea_reranker_lab.combiner import combiner_columns_for_category  # noqa: E402
from protea_reranker_lab.evaluate import (  # noqa: E402
    BandArtifacts,
    EvalOptions,
    PredictionArrays,
    eval_f_micro_w_from_arrays,
)
from protea_reranker_lab.ia_weighting import ia_weights, load_ia_table  # noqa: E402

CATEGORIES = ("nk", "lk", "pk")
ASPECTS = ("mfo", "bpo", "cco")
# Columns that exist in the export but are zero-filled (compute flags off).
ZERO_COLS = frozenset({"association_total", "association_cross", "classifier_score"})
# GO namespace roots (no_orphans drops these from scoring).
GO_ROOTS = frozenset({"GO:0008150", "GO:0003674", "GO:0005575"})
RAW_KNN_COL = "real_knn"  # synthesized below from neighbor_vote_fraction
HOLDOUT_PAIR = "v220-v225"


# ---------------------------------------------------------------------------
# Ontology (parse the band-congruent OBO; no DB access)
# ---------------------------------------------------------------------------
def parse_obo_parents(obo_path: Path) -> dict[str, frozenset[str]]:
    """Parse is_a + part_of parents from a GO OBO file -> {child: {parents}}."""
    parents: dict[str, set[str]] = {}
    cur: str | None = None
    with obo_path.open() as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line == "[Term]":
                cur = None
            elif line.startswith("id: GO:"):
                cur = line[4:].strip()
                parents.setdefault(cur, set())
            elif cur is not None and line.startswith("is_a: GO:"):
                parents[cur].add(line[6:].split("!")[0].strip())
            elif cur is not None and line.startswith("relationship: part_of GO:"):
                parents[cur].add(line.split("part_of", 1)[1].split("!")[0].strip())
    return {c: frozenset(p) for c, p in parents.items()}


def ancestor_resolver(parent_map: dict[str, frozenset[str]]):
    """Return a memoised ancestors(term) -> frozenset (excludes the term)."""
    cache: dict[str, frozenset[str]] = {}

    def ancestors(go_id: str) -> frozenset[str]:
        hit = cache.get(go_id)
        if hit is not None:
            return hit
        seen: set[str] = set()
        stack = list(parent_map.get(go_id, ()))
        while stack:
            a = stack.pop()
            if a not in seen:
                seen.add(a)
                stack.extend(parent_map.get(a, ()))
        res = frozenset(seen)
        cache[go_id] = res
        return res

    return ancestors


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def category_feature_cols(cat: str) -> list[str]:
    """Per-category combiner columns, minus the zero-filled scorers."""
    return [c for c in combiner_columns_for_category(cat) if c not in ZERO_COLS]


def _load_train(train_pq: Path, feat_union: list[str]):
    cols = sorted(
        set(feat_union)
        | {"label", "category", "protein_accession", "go_term_id", "snapshot_pair"}
    )
    return pq.read_table(str(train_pq), columns=cols).to_pandas()


def _groups(prot: np.ndarray) -> np.ndarray:
    """Contiguous run-length group sizes for a protein-sorted array."""
    if prot.size == 0:
        return np.empty(0, dtype=np.int32)
    edges = np.flatnonzero(np.concatenate(([True], prot[1:] != prot[:-1])))
    return np.diff(np.concatenate((edges, [prot.size]))).astype(np.int32)


def train_category(
    df, cat: str, ia_table: dict[str, float], rounds: int, early: int, mlflow_mod
):
    """Train one per-category lambdarank booster; return (booster, cols, info)."""
    cols = category_feature_cols(cat)
    sub = df[df["category"] == cat]
    tr = sub[sub["snapshot_pair"] != HOLDOUT_PAIR].sort_values("protein_accession")
    va = sub[sub["snapshot_pair"] == HOLDOUT_PAIR].sort_values("protein_accession")

    x_tr = tr[cols].to_numpy(dtype=np.float32)
    y_tr = tr["label"].to_numpy(dtype=np.int8)
    w_tr = ia_weights(
        tr["go_term_id"].to_numpy(), y_tr, ia_table, mode="all"
    )
    g_tr = _groups(tr["protein_accession"].to_numpy())

    d_tr = lgb.Dataset(x_tr, label=y_tr, weight=w_tr, group=g_tr, feature_name=cols)
    valid, names, callbacks = [], [], [lgb.log_evaluation(period=100)]
    if len(va):
        x_va = va[cols].to_numpy(dtype=np.float32)
        y_va = va["label"].to_numpy(dtype=np.int8)
        g_va = _groups(va["protein_accession"].to_numpy())
        d_va = lgb.Dataset(x_va, label=y_va, group=g_va, reference=d_tr, feature_name=cols)
        valid, names = [d_va], ["holdout"]
        callbacks.append(lgb.early_stopping(early, verbose=False))
    if mlflow_mod is not None:
        callbacks.append(_mlflow_cb(mlflow_mod, cat))

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5, 10, 30],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "label_gain": [0, 1],
        "verbosity": -1,
    }
    booster = lgb.train(
        params, d_tr, num_boost_round=rounds,
        valid_sets=valid, valid_names=names, callbacks=callbacks,
    )
    info = {
        "train_rows": int(len(y_tr)),
        "train_pos": int(y_tr.sum()),
        "holdout_rows": int(len(va)),
        "best_iter": int(booster.best_iteration or rounds),
        "features": cols,
    }
    return booster, cols, info


def _mlflow_cb(mlflow_mod, cat: str):
    def _cb(env) -> None:
        try:
            for ds, metric, value, _ in env.evaluation_result_list:
                mlflow_mod.log_metric(f"{cat}_{ds}_{metric}", value, step=env.iteration)
        except Exception:
            pass
    return _cb


# ---------------------------------------------------------------------------
# IA-weighted micro-F (numpy reimplementation of cafaeval f_micro_w)
# ---------------------------------------------------------------------------
def build_components(
    proteins, go_terms, scores, labels, ancestors, ia_table, n_thresh, max_terms,
    grid=None,
):
    """Per-protein propagated IA-weighted TP/FP step components over a threshold grid.

    Returns (WTP, WFP, WFN, thresholds) with WTP/WFP/WFN shape (n_prot, n_thresh).
    Mirrors cafaeval: prop='fill' (ancestor score = max descendant), no_orphans
    (drop GO roots), max_terms cap per protein. When ``grid`` is given (e.g.
    cafaeval's fixed ``np.arange(th_step, 1, th_step)``) it is used verbatim so the
    bootstrap mirrors cafaeval's scale-sensitive sweep; otherwise a per-cell
    quantile grid (scale-invariant) is built.
    """
    order = np.argsort(proteins, kind="stable")
    proteins, go_terms = proteins[order], go_terms[order]
    scores, labels = scores[order].astype(np.float64), labels[order]
    sizes = _groups(proteins)
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))

    ent_prot: list[int] = []
    ent_score: list[float] = []
    ent_ia: list[float] = []
    ent_true: list[bool] = []
    wia_true = np.zeros(sizes.size, dtype=np.float64)

    for pi, (s0, sz) in enumerate(zip(starts, sizes)):
        sl = slice(s0, s0 + sz)
        terms = go_terms[sl]
        sc = scores[sl]
        lab = labels[sl]
        # max_terms cap by score (cafaeval caps the prediction file per protein)
        if sz > max_terms:
            keep = np.argsort(-sc)[:max_terms]
            terms, sc, lab = terms[keep], sc[keep], lab[keep]
        # propagate predicted scores up the DAG (fill = max)
        pred: dict[str, float] = {}
        for t, v in zip(terms, sc):
            if t not in GO_ROOTS:
                if v > pred.get(t, -np.inf):
                    pred[t] = v
            for a in ancestors(t):
                if a in GO_ROOTS:
                    continue
                if v > pred.get(a, -np.inf):
                    pred[a] = v
        # propagate true terms
        true_set: set[str] = set()
        for t in terms[lab > 0]:
            if t not in GO_ROOTS:
                true_set.add(t)
            true_set.update(a for a in ancestors(t) if a not in GO_ROOTS)
        wia_true[pi] = float(sum(ia_table.get(t, 0.0) for t in true_set))
        for t, v in pred.items():
            ia = ia_table.get(t, 0.0)
            if ia <= 0.0:
                continue
            ent_prot.append(pi)
            ent_score.append(v)
            ent_ia.append(ia)
            ent_true.append(t in true_set)

    n_prot = sizes.size
    ep = np.asarray(ent_prot, dtype=np.int64)
    es = np.asarray(ent_score, dtype=np.float64)
    eia = np.asarray(ent_ia, dtype=np.float64)
    et = np.asarray(ent_true, dtype=bool)

    if es.size == 0:
        mm = len(grid) if grid is not None else n_thresh
        z = np.zeros((n_prot, mm))
        return z, z.copy(), np.tile(wia_true[:, None], (1, mm)), np.zeros(mm)

    if grid is not None:
        thresholds = np.asarray(grid, dtype=np.float64)
    else:
        thresholds = np.quantile(es, np.linspace(0.0, 1.0, n_thresh))
        thresholds = np.unique(thresholds)
    m = thresholds.size
    # entry active for T[j] <= score: active indices 0..k-1, k = searchsorted right
    k = np.searchsorted(thresholds, es, side="right")
    v_tp = np.where(et, eia, 0.0)
    v_fp = np.where(et, 0.0, eia)
    d_tp = np.zeros((n_prot, m + 1))
    d_fp = np.zeros((n_prot, m + 1))
    np.add.at(d_tp, (ep, np.zeros_like(ep)), v_tp)
    np.add.at(d_tp, (ep, k), -v_tp)
    np.add.at(d_fp, (ep, np.zeros_like(ep)), v_fp)
    np.add.at(d_fp, (ep, k), -v_fp)
    wtp = np.cumsum(d_tp, axis=1)[:, :m]
    wfp = np.cumsum(d_fp, axis=1)[:, :m]
    wfn = wia_true[:, None] - wtp
    return wtp, wfp, wfn, thresholds


def fmicrow_from_components(wtp, wfp, wfn, idx=None):
    """Pooled IA-weighted micro-F across a threshold grid; return (fmax, P, R, j*)."""
    if idx is not None:
        wtp, wfp, wfn = wtp[idx], wfp[idx], wfn[idx]
    tp = wtp.sum(0)
    fp = wfp.sum(0)
    fn = wfn.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        r = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    j = int(np.argmax(f))
    return float(f[j]), float(p[j]), float(r[j]), j


def bootstrap_delta(wtp_a, wfp_a, wfn_a, wtp_b, wfp_b, wfn_b, n_iter, seed):
    """Paired protein bootstrap of (fmw_b - fmw_a). a=baseline, b=reranked."""
    n = wtp_a.shape[0]
    rng = np.random.default_rng(seed)
    da = np.empty(n_iter)
    fa = np.empty(n_iter)
    fb = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        a = fmicrow_from_components(wtp_a, wfp_a, wfn_a, idx)[0]
        b = fmicrow_from_components(wtp_b, wfp_b, wfn_b, idx)[0]
        fa[i], fb[i], da[i] = a, b, b - a
    return {
        "delta_mean": float(da.mean()),
        "delta_ci_lo": float(np.quantile(da, 0.025)),
        "delta_ci_hi": float(np.quantile(da, 0.975)),
        "p_delta_le_0": float((da <= 0).mean()),
        "raw_ci_lo": float(np.quantile(fa, 0.025)),
        "raw_ci_hi": float(np.quantile(fa, 0.975)),
        "rerank_ci_lo": float(np.quantile(fb, 0.025)),
        "rerank_ci_hi": float(np.quantile(fb, 0.975)),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--obo", type=Path, required=True)
    ap.add_argument("--ia", type=Path, required=True)
    ap.add_argument("--num-boost-round", type=int, default=2000)
    ap.add_argument("--early-stopping-rounds", type=int, default=80)
    ap.add_argument("--n-thresh", type=int, default=400)
    ap.add_argument("--max-terms", type=int, default=500)
    ap.add_argument("--n-iter", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--protea-python", type=Path, default=None)
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_pq = args.dataset_dir / "train.parquet"
    eval_pq = args.dataset_dir / "eval.parquet"
    artifacts = BandArtifacts(obo_path=args.obo, ia_path=args.ia)
    opts = EvalOptions(protea_python=args.protea_python, timeout=1800)

    print("[setup] parsing OBO + IA ...", flush=True)
    parent_map = parse_obo_parents(args.obo)
    ancestors = ancestor_resolver(parent_map)
    ia_table = load_ia_table(args.ia)
    print(f"[setup] obo terms={len(parent_map)} ia terms={len(ia_table)}", flush=True)

    mlflow_mod = None
    if not args.no_mlflow:
        try:
            import mlflow
            os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
            os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
            os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")
            mlflow.set_tracking_uri(
                os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
            )
            mlflow.set_experiment("reranker-clean-A4-train225")
            mlflow.start_run(run_name="per-category-lambdamart")
            mlflow.log_params({
                "dataset": "fullgo-clean-A4-train225-val",
                "objective": "lambdarank", "ia_weighting": "all",
                "num_boost_round": args.num_boost_round,
                "train_holdout": HOLDOUT_PAIR, "eval_pair": "v225-v227",
            })
            mlflow_mod = mlflow
        except Exception as exc:
            print(f"[mlflow] disabled: {exc}", flush=True)

    feat_union = sorted({c for cat in CATEGORIES for c in category_feature_cols(cat)})
    print(f"[train] loading train.parquet cols={feat_union} ...", flush=True)
    df = _load_train(train_pq, feat_union)
    print(f"[train] loaded {len(df):,} rows", flush=True)

    boosters: dict[str, lgb.Booster] = {}
    cols_by_cat: dict[str, list[str]] = {}
    train_info: dict[str, dict] = {}
    for cat in CATEGORIES:
        print(f"[train] category {cat} cols={category_feature_cols(cat)}", flush=True)
        b, cols, info = train_category(
            df, cat, ia_table, args.num_boost_round, args.early_stopping_rounds, mlflow_mod
        )
        boosters[cat], cols_by_cat[cat], train_info[cat] = b, cols, info
        print(f"[train] {cat}: rows={info['train_rows']:,} pos={info['train_pos']:,} "
              f"best_iter={info['best_iter']}", flush=True)
    del df

    print("[eval] loading eval.parquet ...", flush=True)
    eval_cols = sorted(
        set(feat_union)
        | {"label", "category", "aspect", "protein_accession", "go_term_id",
           "neighbor_vote_fraction"}
    )
    edf = pq.read_table(str(eval_pq), columns=eval_cols).to_pandas()
    print(f"[eval] loaded {len(edf):,} rows", flush=True)

    results: list[dict] = []
    for cat in CATEGORIES:
        for asp in ASPECTS:
            cell = f"{cat}-{asp}"
            cdf = edf[(edf["category"] == cat) & (edf["aspect"] == asp)]
            if len(cdf) == 0:
                continue
            prot = cdf["protein_accession"].to_numpy()
            go = cdf["go_term_id"].to_numpy()
            lab = cdf["label"].to_numpy(dtype=np.int8)
            raw = cdf["neighbor_vote_fraction"].to_numpy(dtype=np.float64)
            rer = boosters[cat].predict(
                cdf[cols_by_cat[cat]].to_numpy(dtype=np.float32)
            ).astype(np.float64)

            # cafaeval point estimates (authoritative)
            ce_raw = eval_f_micro_w_from_arrays(
                PredictionArrays(prot, go, raw, lab), artifacts, asp, options=opts
            )
            ce_rer = eval_f_micro_w_from_arrays(
                PredictionArrays(prot, go, rer, lab), artifacts, asp, options=opts
            )
            fmw_raw = ce_raw.get("f_micro_w")
            fmw_rer = ce_rer.get("f_micro_w")

            # numpy components for CI + PK precision/recall split
            wtp_a, wfp_a, wfn_a, _ = build_components(
                prot, go, raw, lab, ancestors, ia_table, args.n_thresh, args.max_terms
            )
            wtp_b, wfp_b, wfn_b, _ = build_components(
                prot, go, rer, lab, ancestors, ia_table, args.n_thresh, args.max_terms
            )
            np_raw = fmicrow_from_components(wtp_a, wfp_a, wfn_a)
            np_rer = fmicrow_from_components(wtp_b, wfp_b, wfn_b)
            boot = bootstrap_delta(
                wtp_a, wfp_a, wfn_a, wtp_b, wfp_b, wfn_b, args.n_iter, args.seed
            )

            row = {
                "cell": cell, "category": cat, "aspect": asp,
                "n_rows": int(len(cdf)),
                "n_proteins": int(wtp_a.shape[0]),
                "n_pos": int((lab > 0).sum()),
                "cafaeval_fmw_raw": fmw_raw,
                "cafaeval_fmw_rerank": fmw_rer,
                "cafaeval_delta": (
                    None if (fmw_raw is None or fmw_rer is None)
                    else round(fmw_rer - fmw_raw, 5)
                ),
                "np_fmw_raw": round(np_raw[0], 5),
                "np_fmw_rerank": round(np_rer[0], 5),
                "np_delta": round(np_rer[0] - np_raw[0], 5),
                "raw_precision_w": round(np_raw[1], 5),
                "raw_recall_w": round(np_raw[2], 5),
                "rerank_precision_w": round(np_rer[1], 5),
                "rerank_recall_w": round(np_rer[2], 5),
                "validation_gap_raw": (
                    None if fmw_raw is None else round(np_raw[0] - fmw_raw, 5)
                ),
                "validation_gap_rerank": (
                    None if fmw_rer is None else round(np_rer[0] - fmw_rer, 5)
                ),
                **{k: round(v, 5) for k, v in boot.items()},
            }
            results.append(row)
            print(
                f"[cell] {cell}: cafaeval raw={fmw_raw} rerank={fmw_rer} "
                f"delta={row['cafaeval_delta']} | boot delta="
                f"{row['delta_mean']} [{row['delta_ci_lo']},{row['delta_ci_hi']}]",
                flush=True,
            )
            if mlflow_mod is not None:
                try:
                    mlflow_mod.log_metrics({
                        f"{cell}_fmw_raw": fmw_raw or 0.0,
                        f"{cell}_fmw_rerank": fmw_rer or 0.0,
                        f"{cell}_delta": row["cafaeval_delta"] or 0.0,
                    })
                except Exception:
                    pass

    summary = {
        "dataset": "fullgo-clean-A4-train225-val",
        "schema_sha": "775611822dd9",
        "train_pairs": "v160-v165 .. v220-v225 (early-stop holdout: v220-v225)",
        "eval_pair": "v225-v227",
        "eval_band": "v227",
        "objective": "lambdarank (IA-weighted, mode=all)",
        "raw_knn_baseline": "neighbor_vote_fraction",
        "dropped_zero_cols": sorted(ZERO_COLS),
        "note_anc2vec": "anc2vec_neighbor_maxcos is constant (1.0) in this export; inert.",
        "n_iter": args.n_iter,
        "train_info": train_info,
        "cells": results,
    }
    deltas = [r["cafaeval_delta"] for r in results if r["cafaeval_delta"] is not None]
    summary["mean_cafaeval_delta"] = (
        round(float(np.mean(deltas)), 5) if deltas else None
    )
    out_json = args.out_dir / "results.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[done] mean cafaeval delta = {summary['mean_cafaeval_delta']}", flush=True)
    print(f"[done] -> {out_json}", flush=True)
    if mlflow_mod is not None:
        try:
            mlflow_mod.log_metric("mean_cafaeval_delta", summary["mean_cafaeval_delta"] or 0.0)
            mlflow_mod.log_artifact(str(out_json))
            mlflow_mod.end_run()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
