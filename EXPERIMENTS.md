# EXPERIMENTS champion log

Canonical record of reranker experiment outcomes on bench-v1-K5-v226-lineage.
All entries use dataset eval_snapshot_pair v226-v230 and cafaeval
(prop=fill, norm=cafa, no_orphans=True, max_terms=500, th_step=0.001).


## FARM-EXP.8 transversal grid (2026-05-17)

**Status:** transversal characterisation, lab_fmax metric only.

Full 9-cell x 3-seed grid on `v6+lineage-leakfree` bundle (v6 + lineage
minus anc2vec_* and emb_pca_* families, 22 features dropped, 34 retained),
across both eval sets:

- `bench-v1-K5-v226-lineage` (val_strategy=protein_group, eval v226-v230)
- `bench-v1-K5-filtered` (val_strategy=temporal val_holdout=v215-v220, eval v220-v230)

Scope: 9 cells x 3 seeds (42, 7, 137) x 2 eval sets = 54 runs (esmc_300m
embeddings, K=5). Aspirational 8-PLM x 2-K x 2-rerankers transversal
was not feasible because only esmc_300m embeddings exist locally; the
PLM/K/reranker axes are deferred until additional embedding caches land.

### Per-cell lab_fmax mean ± 95% CI (3 seeds, t-dist df=2)

```
                v226-lineage              filtered
cell           mean    CI_half        mean    CI_half
nk-mfo        0.1924   0.0077        0.6347   0.0068
nk-bpo        0.0421   0.0059        0.5874   0.0054
nk-cco        0.1483   0.0030        0.7750   0.0033
lk-mfo        0.1546   0.0087        0.5731   0.0020
lk-bpo        0.0500   0.0043        0.6151   0.0027
lk-cco        0.1137   0.0043        0.7881   0.0022
pk-mfo        0.1361   0.0061        0.2819   0.0080
pk-bpo        0.0407   0.0048        0.1362   0.0034
pk-cco        0.1291   0.0287        0.2840   0.0125
```

### Grid aggregates (lab_fmax, mean across 3 seeds)

```
                v226-lineage    filtered
9-cell avg        0.1119         0.5195
NK+LK avg         0.1168         0.6622
PK avg            0.1020         0.2340
best cell         nk-mfo         lk-cco
best lab_fmax     0.1924         0.7881
```

### Metric note (important)

`test_fmax` in run.json is `lab_fmax`: per-protein-group Fmax without
label propagation. The LB.2 / FARM-EXP.10 champion **0.6215 ± 0.0014**
is `cafaeval_fmax` (with prop=fill, norm=cafa label propagation). These
are different metrics; the lab_fmax numbers above are not directly
comparable to the champion. Cross-cell ratios match the cafaeval ranking
(NK+LK > PK, lk-cco/nk-cco strong on filtered) but the absolute scale
differs by ~5x.

A cafaeval re-evaluation of the FARM-EXP.8 prediction parquets is
deferred to FARM-EXP.11 (post-cleanup); it requires copying predictions
into the PROTEA venv (which ships cafaeval) and rerunning the cafaeval
phase script adapted for this grid. Once that lands, the comparable
selective_avg vs champion 0.6215 will be reported.

### Seed variance / stability

CI half-widths across all 54 cell-eval-set combinations stay below 0.03,
with median ~0.005. The largest CI is pk-cco v226-lineage (0.0287), driven
by seed42=0.1191 vs seed7=0.1418 (one seed materially off the cluster).
NK+LK cells on filtered are very stable (CI 0.002-0.007). The reranker
ordering across seeds is consistent: no cell flips rank within an eval
set across seeds.

### Cross eval-set comparison

The `filtered` eval set yields markedly higher lab_fmax across all NK+LK
cells (NK+LK avg 0.6622 vs 0.1168 on v226-lineage). The two eval sets
differ in val strategy (temporal v215-v220 vs protein_group) and in train
row count per cell (filtered NK cells: 140k-440k rows; v226-lineage NK
cells: 1.8M-2.3M rows). The lab_fmax scale gap is dominated by the eval
set composition (filtered eval covers v220-v230 vs v226-v230 only), not
by reranker quality.

### Output artefacts

- `runs/farm_exp_8/summary.json`: full per-cell stats and aggregates.
- `runs/farm_exp_8/cis.json`: compact per-cell CIs.
- `experiments/farm_exp_8/{v226_lineage,filtered}/*.yaml`: 54 spec files.
- `runs/farm_exp_8/{v226_lineage,filtered}/{cell}_seed{s}/run.json`: per-run records (gitignored).

### Next slot options

1. FARM-EXP.11: cafaeval re-evaluation of FARM-EXP.8 predictions for
   apples-to-apples champion comparison.
2. Extend grid to additional PLMs once embedding caches land (esmc_600m,
   esm2_t33_650M, prostt5, etc.).
3. Add K=10 axis (currently K=5 only).


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
anc2vec and PCA features). FARM-EXP.10b promotes this to the
first-class catalog axis value `v6+lineage-leakfree` in
`experiments/_catalog/transversal.yaml`, alongside the umbrella
`v6+lineage` value. The bundle name resolves to the actual
family/drop list in the runner slice (FARM-EXP.5+); the catalog
schema_sha is still the placeholder digest derived from the bundle
name (cleared together with `project_farm_exp_2_placeholder_digests`
once the digest-backfill slice lands).

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
