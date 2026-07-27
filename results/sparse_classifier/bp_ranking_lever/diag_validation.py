"""Validation signal probe: do term-space features separate true vs false BP terms?"""
import os, sys, json, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from features import build_term_features

HERE = os.path.dirname(os.path.abspath(__file__))


def pminmax_bpo(df):
    """Per-protein min-max of base_score (matches LK-bpo serve lever)."""
    g = df.groupby("protein_accession")["base_score"]
    mn = g.transform("min"); mx = g.transform("max"); rng = mx - mn
    df["base_score"] = np.where(rng > 0, (df["base_score"] - mn) / rng, 1.0)
    return df


def get_valid_base(parquet, model_path, category, apply_pminmax):
    df, feats = C.load_valid_bp(parquet, category)
    booster = lgb.Booster(model_file=model_path)
    df["base_score"] = booster.predict(df[feats], num_iteration=booster.best_iteration)
    keep = df[["protein_accession", "go_term_id", "label", "base_score"]].copy()
    if apply_pminmax:
        keep = pminmax_bpo(keep)
    return keep


def main():
    print("loading resources...")
    ns_map = C.load_obo_namespace()
    ia = C.load_ia()
    code_idx, codes = C.load_codes()
    depth = C.compute_depths()

    for cat, parquet, model, pmm in [
        ("lk", C.LK_TRAIN_PARQUET, C.LK_MODEL, True),
        ("pk", C.PK_TRAIN_PARQUET, C.PK_MODEL, False),
    ]:
        t0 = time.time()
        base = get_valid_base(parquet, model, cat, pmm)
        # build features needs all-aspect candidates for the profile, but validation
        # parquet here is BP-only filtered. Use BP candidates as the profile context.
        feat = build_term_features(
            base.rename(columns={}), ns_map, code_idx, codes, depth, ia, topk=10)
        # NB: build_term_features filters to BP namespace; base is already BP rows.
        y = feat["label"].values.astype(int)
        print(f"\n=== {cat.upper()}-bpo validation: {len(feat)} rows pos={y.sum()} "
              f"code_cov={feat['has_code'].mean():.3f} ({time.time()-t0:.1f}s) ===")
        for col in ["base_score", "profile_fit", "topk_coherence", "depth", "ia"]:
            try:
                auc = roc_auc_score(y, feat[col].values)
            except Exception:
                auc = float("nan")
            pos_m = feat.loc[feat.label == 1, col].mean()
            neg_m = feat.loc[feat.label == 0, col].mean()
            print(f"  {col:16s} AUC={auc:.4f}  pos_mean={pos_m:.4f} neg_mean={neg_m:.4f}")
        feat.to_parquet(os.path.join(HERE, f"valid_{cat}_bp_feat.parquet"))


if __name__ == "__main__":
    main()
