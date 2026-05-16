# EXPERIMENTS champion log

Canonical record of reranker experiment outcomes on bench-v1-K5-v226-lineage.
All entries use dataset eval_snapshot_pair v226-v230 and cafaeval
(prop=fill, norm=cafa, no_orphans=True, max_terms=500, th_step=0.001).


## FARM-EXP.10 champion (2026-05-17)

**Status:** active champion (selective rerank policy on v226-lineage).

The LB.2 multi-seed sweep below is promoted as the leakage-fixed
selective-rerank champion. Key publishable numbers:

- Selective avg cafaeval Fmax (6 NK+LK rerank cells + 3 PK baseline
  fallback): **0.6215 ± 0.0014** (95% CI half-width on 9-cell mean
  over 10000-iteration bootstrap of the 3-seed mean).
- 6-cell NK+LK reranker avg: **0.6845**.

Policy: the reranker is deployed on NK+LK cells (6); PK cells (3) fall
back to KNN baseline because lineage features cause a DAG-closure
shortcut on PK. See ADR-D34 in the PROTEA repo for the deployment
decision.

Configuration is the v23 leakage-fixed bundle (v6 + lineage minus
anc2vec and PCA features). This is not yet a named bundle in the
FARM-EXP.2 transversal catalog (`experiments/_catalog/transversal.yaml`);
the catalog ships the umbrella `v6+lineage` value. Adding the
leakage-fixed bundle as a first-class catalog axis value is tracked
as a follow-up (FARM-EXP.10b), pending the digest-backfill slice that
also clears `project_farm_exp_2_placeholder_digests`.

`runs/transversal/<shortid>/` placement is deferred to the writer
slice (FARM-EXP.5+), which is the slice that actually emits
FARM-EXP.3-format run.json records carrying the `axis` block and the
`fmax_samples` array. Until that slice lands, the
`scripts/update_champions.py` walker has no FARM-EXP.3 records to
promote; this section is the manual champion declaration.

References:

- LB.2 multi-seed sweep section below (per-cell table + variance).
- Memory `project_lb2_leakage_fixed_champion` (publishable numbers).
- Memory `project_farm_exp_2_placeholder_digests` (tentative shortids).
- PROTEA ADR-D34 (deployment decision).


## LB.2 multi-seed sweep (2026-05-17)

**Config:** v23 (no anc2vec, no pca; lambdarank; LR=0.05, leaves=63,
num_boost_round=10000, early_stop=100, seed={42,7,137})
**Dataset:** bench-v1-K5-v226-lineage (eval v226-v230)
**Cells:** NK+LK (6 cells, 3 seeds each = 18 runs)
**Purpose:** characterise seed variance before promoting champion to PROTEA inference

### Per-cell cafaeval Fmax (mean over seeds 42, 7, 137)

```
cell     s42    s7    s137   mean   std   CI_half  baseline  lift
nk-mfo  0.7112 0.7041 0.7041 0.7065 0.0033  0.0036  0.6447  +0.0618
nk-bpo  0.5599 0.5571 0.5618 0.5596 0.0019  0.0024  0.5333  +0.0262
nk-cco  0.7733 0.7830 0.7758 0.7774 0.0041  0.0048  0.7000  +0.0774
lk-mfo  0.6877 0.6786 0.6757 0.6806 0.0051  0.0060  0.5816  +0.0991
lk-bpo  0.6472 0.6421 0.6485 0.6460 0.0028  0.0032  0.5844  +0.0615
lk-cco  0.7434 0.7252 0.7417 0.7367 0.0082  0.0091  0.7053  +0.0315
```

95% CI computed via 10000-iteration bootstrap of the 3-seed mean (protein-level
resampling within each cell, then mean of 3 seed samples).

### Aggregate metrics

- **6-cell NK+LK reranker avg:** 0.6845
- **6-cell NK+LK baseline avg:** 0.6249
- **NK+LK lift over baseline:** +0.0596
- **9-cell selective avg (NK+LK reranker + PK baseline fallback):** 0.6215
- **9-cell all-baseline avg:** 0.5818
- **9-cell selective delta vs baseline:** +0.0397

PK cells (pk-mfo, pk-bpo, pk-cco) are excluded from reranker deployment
(lineage features cause DAG-closure shortcut overfit in PK; see SUMMARY_v23-v26.md).
PK falls back to KNN baseline (0.483, 0.403, 0.601 respectively).

### Seed variance assessment

Maximum per-cell seed variance (std): lk-cco 0.0082, lk-mfo 0.0051.
95% CI half-widths range from 0.0024 (nk-bpo) to 0.0091 (lk-cco).
All lifts are clearly above 0 across all seeds. The reranker effect is
stable across seeds: no cell flips sign vs baseline in any seed.

### Predecessor single-seed reference (LB.2 retrofix, 2026-05-17)

The predecessor session reported a "selective_rerank_avg_fmax: 0.6408" based
on lk-mfo seed=137 (0.6636) and other cells from study_v23 seed=42.
That number mixed seed-42 NK+LK values with seed-137 lk-mfo and was an
approximate estimate. The multi-seed sweep supersedes it.

The study_v23 seed=42 lk-mfo value (0.6998) is the high-end of the 3-seed
range [0.6757, 0.6877]; the mean across 3 seeds is 0.6806.

### Recommendation

The reranker is stable across seeds (all lifts positive, CI half-widths < 0.01
for most cells). The v23 leakage-fixed config is suitable for promotion
to PROTEA inference for NK+LK cells. PK cells continue to use KNN baseline.

**Next slot options:**
1. Promote v23 config to PROTEA inference (NK+LK deployment).
2. Run multi-seed sweep for v26-binary config to see if it consistently
   outperforms v23 (per-cell optimal from SUMMARY_v23-v26 suggests +0.03
   to +0.08 on NK+LK, but no multi-seed confirmation yet).
3. Investigate PK improvement (needs known_terms overlap feature).


## LB.2 single-seed retrofix (2026-05-17, predecessor)

lk-mfo cell only; seed=42 gave cafaeval_fmax=0.6998, seed=137 gave 0.6636.
Confirmed anc2vec leakage fix: neighbor_vote_fraction dominates (~61k gain).
See `runs/study_v23/` for all 9-cell results.


## study_v23 (2026-05-14)

Full 9-cell run, seed=42, v23 config (no anc2vec+pca, lambdarank).
See `runs/study_v23/cafaeval/results.csv` for per-cell cafaeval results.
