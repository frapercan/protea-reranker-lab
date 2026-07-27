"""Build the LK union pool with the GCN JOINT SCORE as the candidate score.

Single-variable change vs Phase 0: SAME union pool (Phase 0 union_lk_{train,eval}.parquet
= champion LK pool UNION two-tower BP candidates), but the score the reranker sees on
every BP row is now the GCN joint scorer's compatibility (gcn_scorer.py, per-cut honest),
not the frozen two-tower logit. Every BP candidate -- including clf-only ones the
retrieval pool never surfaces -- gets a MEANINGFUL joint score, which is exactly the
Phase 0 neck (joint scoring).

Overwrites tt_clf_score := gcn_score on BP rows; recomputes tt_clf_rank from gcn_score;
tt_has_clf := 1 where a finite joint score exists. Non-BP rows unchanged (score 0).
Writes union_lk_gcn_{train,eval}.parquet.
"""
import os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE0 = os.path.join(os.path.dirname(HERE), "bp_structural_phase0")
sys.path.insert(0, HERE)
from gcn_scorer import GcnScorer  # noqa: E402


def build(split, scorer):
    src = os.path.join(PHASE0, f"union_lk_{split}.parquet")
    df = pd.read_parquet(src)
    is_bp = (df["aspect"] == "bpo").to_numpy()
    bp = df.loc[is_bp, ["protein_accession", "go_term_id", "snapshot_pair"]].rename(
        columns={"protein_accession": "acc", "go_term_id": "go"})
    t0 = time.time()
    gs = scorer.score_frame(bp)
    print(f"[{split}] scored {len(bp)} BP rows in {time.time()-t0:.1f}s "
          f"(finite {np.isfinite(gs).mean()*100:.1f}%)", flush=True)
    df["tt_clf_score"] = 0.0
    df.loc[is_bp, "tt_clf_score"] = gs
    df["tt_clf_score"] = df["tt_clf_score"].astype("float32")
    df["tt_has_clf"] = 0
    df.loc[is_bp, "tt_has_clf"] = np.isfinite(gs).astype("int8")
    df["tt_has_clf"] = df["tt_has_clf"].astype("int8")
    # within-(protein, snapshot) rank of the joint score over BP candidates
    df["tt_clf_rank"] = np.nan
    bpm = is_bp & np.isfinite(df["tt_clf_score"].to_numpy()) & (df["tt_has_clf"].to_numpy() == 1)
    sub = df.loc[bpm, ["protein_accession", "snapshot_pair", "tt_clf_score"]]
    r = sub.groupby(["protein_accession", "snapshot_pair"])["tt_clf_score"] \
           .rank(ascending=False, method="first")
    df.loc[bpm, "tt_clf_rank"] = r.values
    df["tt_clf_rank"] = df["tt_clf_rank"].astype("float32")
    out = os.path.join(HERE, f"union_lk_gcn_{split}.parquet")
    df.to_parquet(out)
    nbp = int(is_bp.sum())
    print(f"[{split}] LK union+GCN rows={len(df)} BP={nbp} has_clf={int(df['tt_has_clf'].sum())} -> {out}",
          flush=True)


if __name__ == "__main__":
    scorer = GcnScorer()
    for split in ("train", "eval"):
        build(split, scorer)
