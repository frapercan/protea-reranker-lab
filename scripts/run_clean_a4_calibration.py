#!/usr/bin/env python
"""Track C2: attack the C1 PK regression with per-aspect score calibration.

C1 found that the per-category LambdaMART reranker lifts NK strongly but
regresses PK (precision collapse). The suspected cause is that lambdarank emits
an UNCALIBRATED score (arbitrary real range), while cafaeval sweeps a FIXED
threshold grid ``np.arange(th_step, 1, th_step)`` on [0, 1); scores outside that
range cannot be thresholded, capping precision. This script fits per
(category, aspect) calibrators (isotonic and Platt) on the v220-v225 holdout
(NOT the eval frame), maps the reranker score into [0, 1], and re-evaluates.

It answers:
  1. Does calibration recover PK (does pk-* stop regressing)? PK precision
     specifically.
  2. Per-cell 4-way f_micro_w: raw-KNN vs reranked-uncal vs reranked-isotonic vs
     reranked-platt, with paired bootstrap CIs.
  3. Pooled per-aspect (CAFA-style: NK+LK+PK under one threshold) mean f_micro_w:
     champion (raw-KNN) vs reranked-all-calibrated vs the per-category GATED
     config (NK/LK reranked, PK raw-KNN). Calibration is what makes a pooled
     multi-source config thresholdable at all.

Reuses the C1 trainer + numpy IA-weighted micro-F (now driven on cafaeval's
fixed grid so the bootstrap is scale-faithful). Run with the lab venv python.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

import run_clean_a4_reranker as C1  # noqa: E402
from protea_reranker_lab.calibration import (  # noqa: E402
    CalibrationSpec,
    _fit_one_calibrator,
)
from protea_reranker_lab.evaluate import (  # noqa: E402
    BandArtifacts,
    EvalOptions,
    PredictionArrays,
    eval_f_micro_w_from_arrays,
)
from protea_reranker_lab.ia_weighting import load_ia_table  # noqa: E402

CAT, ASP = C1.CATEGORIES, C1.ASPECTS
# cafaeval's fixed sweep (driver uses th_step=0.001).
CAFA_GRID = np.arange(0.001, 1.0, 0.001)


class Calib:
    """A fitted (cat, aspect) calibrator with a clip-to-[0,1] identity fallback."""

    def __init__(self, est, method: str):
        self.est = est
        self.method = method

    def transform(self, s: np.ndarray) -> np.ndarray:
        if self.est is None:
            return np.clip(s.astype(np.float64), 0.0, 1.0)
        if self.method == "platt":
            return self.est.predict_proba(s.reshape(-1, 1).astype(float))[:, 1]
        return self.est.transform(s.astype(float))


def fit_calibrators(scores, labels, method: str) -> Calib:
    spec = CalibrationSpec(method=method)
    est = _fit_one_calibrator(np.asarray(scores), np.asarray(labels), spec)
    return Calib(est, method)


def cafa_fmw(prot, go, score, lab, artifacts, asp, opts) -> float | None:
    return eval_f_micro_w_from_arrays(
        PredictionArrays(prot, go, score.astype(np.float64), lab), artifacts, asp,
        options=opts,
    ).get("f_micro_w")


def components(prot, go, score, lab, anc, ia, max_terms):
    return C1.build_components(
        prot, go, score.astype(np.float64), lab, anc, ia, 0, max_terms, grid=CAFA_GRID
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--obo", type=Path, required=True)
    ap.add_argument("--ia", type=Path, required=True)
    ap.add_argument("--num-boost-round", type=int, default=2000)
    ap.add_argument("--early-stopping-rounds", type=int, default=80)
    ap.add_argument("--max-terms", type=int, default=500)
    ap.add_argument("--n-iter", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--protea-python", type=Path, default=None)
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = BandArtifacts(obo_path=args.obo, ia_path=args.ia)
    opts = EvalOptions(protea_python=args.protea_python, timeout=1800)

    print("[setup] parsing OBO + IA ...", flush=True)
    parent_map = C1.parse_obo_parents(args.obo)
    anc = C1.ancestor_resolver(parent_map)
    ia_table = load_ia_table(args.ia)

    feat_union = sorted({c for cat in CAT for c in C1.category_feature_cols(cat)})
    cols = sorted(
        set(feat_union)
        | {"label", "category", "aspect", "protein_accession", "go_term_id",
           "snapshot_pair", "neighbor_vote_fraction"}
    )
    print("[train] loading train.parquet ...", flush=True)
    df = pq.read_table(str(args.dataset_dir / "train.parquet"), columns=cols).to_pandas()

    # Train per-category boosters (C1) and fit calibrators on the holdout.
    boosters, cols_by_cat, calib = {}, {}, {}
    for cat in CAT:
        b, cc, info = C1.train_category(
            df, cat, ia_table, args.num_boost_round, args.early_stopping_rounds, None
        )
        boosters[cat], cols_by_cat[cat] = b, cc
        print(f"[train] {cat}: rows={info['train_rows']:,} best_iter={info['best_iter']}",
              flush=True)
        ho = df[(df["category"] == cat) & (df["snapshot_pair"] == C1.HOLDOUT_PAIR)]
        raw = b.predict(ho[cc].to_numpy(dtype=np.float32)).astype(np.float64)
        for asp in ASP:
            m = (ho["aspect"] == asp).to_numpy()
            if m.sum() < 50:
                continue
            y = ho["label"].to_numpy(dtype=np.int8)[m]
            calib[(cat, asp, "iso")] = fit_calibrators(raw[m], y, "isotonic")
            calib[(cat, asp, "platt")] = fit_calibrators(raw[m], y, "platt")
            # calibrate the raw-KNN arm too (for the pooled gated PK arm)
            vf = ho["neighbor_vote_fraction"].to_numpy(dtype=np.float64)[m]
            calib[(cat, asp, "knn_iso")] = fit_calibrators(vf, y, "isotonic")
    del df

    print("[eval] loading eval.parquet ...", flush=True)
    edf = pq.read_table(str(args.dataset_dir / "eval.parquet"), columns=cols).to_pandas()

    # --- per-cell 4-way: raw / uncal / iso / platt ---
    cells, pooled_store = [], {}
    for cat in CAT:
        for asp in ASP:
            c = edf[(edf["category"] == cat) & (edf["aspect"] == asp)]
            if len(c) == 0:
                continue
            prot = c["protein_accession"].to_numpy()
            go = c["go_term_id"].to_numpy()
            lab = c["label"].to_numpy(dtype=np.int8)
            raw = c["neighbor_vote_fraction"].to_numpy(dtype=np.float64)
            uncal = boosters[cat].predict(
                c[cols_by_cat[cat]].to_numpy(dtype=np.float32)
            ).astype(np.float64)
            iso = calib[(cat, asp, "iso")].transform(uncal)
            platt = calib[(cat, asp, "platt")].transform(uncal)

            variants = {"raw": raw, "uncal": uncal, "iso": iso, "platt": platt}
            ce = {k: cafa_fmw(prot, go, v, lab, artifacts, asp, opts)
                  for k, v in variants.items()}
            comp = {k: components(prot, go, v, lab, anc, ia_table, args.max_terms)
                    for k, v in variants.items()}
            pt = {k: C1.fmicrow_from_components(*comp[k][:3]) for k in variants}
            boot_iso = C1.bootstrap_delta(
                *comp["raw"][:3], *comp["iso"][:3], args.n_iter, args.seed
            )
            boot_uncal = C1.bootstrap_delta(
                *comp["raw"][:3], *comp["uncal"][:3], args.n_iter, args.seed
            )

            row = {
                "cell": f"{cat}-{asp}", "category": cat, "aspect": asp,
                "n_proteins": int(comp["raw"][0].shape[0]),
                "n_pos": int((lab > 0).sum()),
                "cafaeval": {k: ce[k] for k in variants},
                "cafaeval_delta_iso_vs_raw": (
                    None if ce["iso"] is None or ce["raw"] is None
                    else round(ce["iso"] - ce["raw"], 5)
                ),
                "cafaeval_delta_uncal_vs_raw": (
                    None if ce["uncal"] is None or ce["raw"] is None
                    else round(ce["uncal"] - ce["raw"], 5)
                ),
                "boot_iso_vs_raw": {k: round(v, 5) for k, v in boot_iso.items()},
                "boot_uncal_vs_raw": {k: round(v, 5) for k, v in boot_uncal.items()},
                "precision_recall_w": {
                    k: {"P": round(pt[k][1], 5), "R": round(pt[k][2], 5)}
                    for k in variants
                },
            }
            cells.append(row)
            # stash per-cell arrays for the pooled per-aspect eval
            pooled_store[(cat, asp)] = {
                "prot": prot, "go": go, "lab": lab, "raw": raw,
                "iso": iso, "knn_iso": calib[(cat, asp, "knn_iso")].transform(raw),
            }
            print(f"[cell] {cat}-{asp}: raw={ce['raw']} uncal={ce['uncal']} "
                  f"iso={ce['iso']} platt={ce['platt']}", flush=True)

    # --- pooled per-aspect (NK+LK+PK under one threshold) ---
    pooled = []
    for asp in ASP:
        keys = [(cat, asp) for cat in CAT if (cat, asp) in pooled_store]
        if not keys:
            continue
        prot = np.concatenate([pooled_store[k]["prot"] for k in keys])
        go = np.concatenate([pooled_store[k]["go"] for k in keys])
        lab = np.concatenate([pooled_store[k]["lab"] for k in keys])
        champ = np.concatenate([pooled_store[k]["raw"] for k in keys])
        rerank_all = np.concatenate([pooled_store[k]["iso"] for k in keys])
        gated = np.concatenate([
            pooled_store[k]["iso"] if k[0] in ("nk", "lk")
            else pooled_store[k]["knn_iso"]
            for k in keys
        ])
        configs = {"champion": champ, "rerank_all_iso": rerank_all, "gated_iso": gated}
        pooled.append({
            "aspect": asp,
            "n_proteins": int(np.unique(prot).size),
            "f_micro_w": {
                name: cafa_fmw(prot, go, s, lab, artifacts, asp, opts)
                for name, s in configs.items()
            },
        })
        print(f"[pooled] {asp}: {pooled[-1]['f_micro_w']}", flush=True)

    def _mean(getter):
        vals = [getter(r) for r in pooled if getter(r) is not None]
        return round(float(np.mean(vals)), 5) if vals else None

    summary = {
        "dataset": "fullgo-clean-A4-train225-val",
        "eval_pair": "v225-v227", "eval_band": "v227",
        "holdout_for_calibration": C1.HOLDOUT_PAIR,
        "cafa_grid": "np.arange(0.001, 1, 0.001) [matches cafaeval driver]",
        "per_cell": cells,
        "pooled_per_aspect": pooled,
        "pooled_means": {
            "champion": _mean(lambda r: r["f_micro_w"]["champion"]),
            "rerank_all_iso": _mean(lambda r: r["f_micro_w"]["rerank_all_iso"]),
            "gated_iso": _mean(lambda r: r["f_micro_w"]["gated_iso"]),
        },
    }
    # per-cell mean deltas (mirrors C1's mean-over-9-cells framing)
    d_iso = [r["cafaeval_delta_iso_vs_raw"] for r in cells
             if r["cafaeval_delta_iso_vs_raw"] is not None]
    d_uncal = [r["cafaeval_delta_uncal_vs_raw"] for r in cells
               if r["cafaeval_delta_uncal_vs_raw"] is not None]
    gate = [
        max(r["cafaeval_delta_iso_vs_raw"], 0.0) if r["category"] in ("nk", "lk")
        else 0.0
        for r in cells if r["cafaeval_delta_iso_vs_raw"] is not None
    ]
    summary["per_cell_mean_delta_iso_vs_raw"] = round(float(np.mean(d_iso)), 5)
    summary["per_cell_mean_delta_uncal_vs_raw"] = round(float(np.mean(d_uncal)), 5)
    summary["per_cell_mean_delta_gated_iso"] = round(float(np.mean(gate)), 5)

    out = args.out_dir / "calibration_results.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[done] per-cell mean delta iso_vs_raw={summary['per_cell_mean_delta_iso_vs_raw']} "
          f"uncal_vs_raw={summary['per_cell_mean_delta_uncal_vs_raw']}", flush=True)
    print(f"[done] pooled means={summary['pooled_means']}", flush=True)
    print(f"[done] -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
