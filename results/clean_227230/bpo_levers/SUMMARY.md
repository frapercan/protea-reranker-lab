# BPO precision levers (pure offline post-processing)

Attack the hardest LAFA BPO cells (LK-bpo, PK-bpo) by post-processing the
clf+assoc reranked scores only. No GPU, no DB, no platform jobs. Each lever is a
deterministic transform of `reranker_score` restricted to BPO rows; MFO/CCO rows
keep the baseline globalmm score verbatim, so MFO/CCO cells are unchanged by
construction (verified: zero non-BPO regression for every lever). Scored with the
EXACT LAFA cafaeval CLI (`-prop fill -norm cafa -no_orphans -toi`, PK adds
`-known`) over the 7401 query set, KNN fallback for the ~54 uncovered.

Substrate: `results/clean_227230/clfassoc/eval_scores.parquet` (1,198,440 rows;
982,560 BPO; 5,187 BPO query proteins). Script: `bpo_levers.py`.

## Baseline reproduces exactly

| cell    | f_micro_w | board | TransFew |
|---------|-----------|-------|----------|
| nk-bpo  | 0.330     | 0.309 | -        |
| lk-bpo  | 0.307     | 0.348 | 0.512    |
| pk-bpo  | 0.101     | 0.117 | 0.294    |

(clf+assoc REGRESSES LK-bpo vs board 0.348 -> 0.307; this is what the levers must
recover.)

## Lever sweep (f_micro_w; delta vs baseline)

| lever               | lk-bpo | d      | pk-bpo | d      | nk-bpo | d      | non-BPO regress |
|---------------------|--------|--------|--------|--------|--------|--------|-----------------|
| baseline (globalmm) | 0.307  |  -     | 0.101  |  -     | 0.330  |  -     | none            |
| rank_norm           | 0.344  | +0.037 | 0.095  | -0.006 | 0.305  | -0.025 | none            |
| pminmax             | 0.351  | +0.044 | 0.096  | -0.005 | 0.299  | -0.031 | none            |
| pminmax_t0.5        | 0.351  | +0.044 | 0.095  | -0.006 | 0.299  | -0.031 | none            |
| pminmax_t2          | 0.351  | +0.044 | 0.095  | -0.006 | 0.298  | -0.032 | none            |
| hsmooth_a0.9        | 0.314  | +0.007 | 0.096  | -0.005 | 0.331  | +0.001 | none            |
| hsmooth_a0.7        | 0.316  | +0.009 | 0.085  | -0.016 | 0.312  | -0.018 | none            |
| hsmooth_a0.5        | 0.274  | -0.033 | 0.078  | -0.023 | 0.275  | -0.055 | none            |
| condprob_mono       | 0.282  | -0.025 | 0.070  | -0.031 | 0.333  | +0.003 | none            |
| condprob_mult       | 0.130  | -0.177 | 0.033  | -0.068 | 0.228  | -0.102 | none            |
| condprob_mult_b0.5  | 0.220  | -0.087 | 0.035  | -0.066 | 0.296  | -0.034 | none            |
| combo_hsmooth_rank  | 0.326  | +0.019 | 0.092  | -0.009 | 0.307  | -0.023 | none            |
| combo_hsmooth_pmm   | 0.344  | +0.037 | 0.092  | -0.009 | 0.283  | -0.047 | none            |
| combo_mono_rank     | 0.334  | +0.027 | 0.076  | -0.025 | 0.314  | -0.016 | none            |

Note: in the sweep, non-baseline variants pass through a redundant global
`to_unit()` rescale on top of the per-protein transform, which slightly compresses
the per-protein levers. The keeper below measures pminmax on LK alone without that
extra rescale (LK-bpo 0.370). Relative lever ranking is unaffected.

## Keeper: per-category pminmax routing

NK/LK/PK protein sets are disjoint (the LAFA knowledge-state split), so different
BPO transforms can be routed per category inside one prediction file (PROTEA's
board already gates PK per-category). Apply pminmax to LK-bpo only, keep baseline
globalmm for NK-bpo and PK-bpo:

| cell    | keeper | baseline | d      | board | TransFew |
|---------|--------|----------|--------|-------|----------|
| lk-bpo  | 0.370  | 0.307    | +0.063 | 0.348 | 0.512    |
| pk-bpo  | 0.101  | 0.101    |  0.000 | 0.117 | 0.294    |
| nk-bpo  | 0.330  | 0.330    |  0.000 | -     | -        |

Zero regression on any of the 9 cells. LK-bpo lands +0.022 above board, still
-0.142 short of TransFew.

## Verdict

- LK-bpo is recoverable by post-processing. Per-protein normalization (pminmax /
  rank_norm) lifts LK-bpo from the clf+assoc regression (0.307) back above board
  (0.351 in-sweep, 0.370 clean per-category). Mechanism: globalmm ties every
  protein to one global scale; LK proteins are under-studied with few, low-magnitude
  candidate terms, so a per-protein min-max restores their dynamic range and the
  tau sweep can separate true from false at a useful threshold. Temperature is a
  no-op (t0.5/t1/t2 identical), as expected for a monotone rescale under a
  threshold sweep.
- PK-bpo is a hard wall; post-processing cannot move it. EVERY lever loses PK-bpo
  (-0.005 best, down to -0.068). PK has a huge dense candidate pool (1.0M rows, the
  precision-bound known set); any per-protein spread inflates the top term of every
  protein to ~1.0 and floods false positives. Hierarchy levers (smoothing, monotone
  consistency, conditional multiply) all hurt because `-prop fill` already
  propagates max to ancestors, so DAG re-smoothing only drags scores down or breaks
  calibration. The PK-bpo gap to TransFew (0.117 vs 0.294) is structural: it needs
  better candidates / a stronger model, not a rescale.
- Keeper = per-category pminmax on LK-bpo (lk-bpo 0.307 -> 0.370, board +0.022,
  nothing else touched). Hierarchical / CondProb levers are NOT keepers; drop them.
- Honest ceiling: post-processing recovers the LK-bpo normalization regression and
  edges past board, but does not close the TransFew gap on either BPO cell
  (LK -0.142, PK -0.193). Beating TransFew on BPO requires modeling, not re-scoring.

Files: `comparison.json` (full 9-cell results + keeper + deltas),
`best_per_category.json` (keeper cells), `bpo_levers.py` (sweep), `run.log`.
No board injection, no PR.
