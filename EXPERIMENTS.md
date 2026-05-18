# EXPERIMENTS champion log

Canonical record of reranker experiment outcomes on bench-v1-K5-v226-lineage.
All entries use dataset eval_snapshot_pair v226-v230 and cafaeval
(prop=fill, norm=cafa, no_orphans=True, max_terms=500, th_step=0.001).


## LB.3 closure (2026-05-18)

**Status:** done. Per-cell paired bootstrap confidence intervals on the
leakage-fixed champion (esmc_300m, K=5, lgbm.per_cell_9, feat=leakage-free+lineage bundle,
eval=bench-v1-K5-v226-lineage, prop=fill, ens=none)
versus the KNN-only baseline, across all 9 cells x 3 aspects. The
selective-deploy policy (NK+LK reranked, PK baseline fallback) is
reflected in the table: PK cells carry zero delta by construction.

### Methodology

Seed-level paired bootstrap (N=10000, seed=42, alpha=0.05, two-sided
95% percentile CI). Data source: the LB.2 multi-seed sweep (3 seeds:
42, 7, 137), which provides per-seed cafaeval Fmax for both the
champion arm and the KNN-only baseline on bench-v1-K5-v226-lineage
(eval v226-v230). Each bootstrap iteration resamples the 3 seed
observations with replacement (paired: same seed-index drawn for both
arms) and records the per-iteration mean Fmax difference. The 2.5th
and 97.5th quantiles of the 10000-iteration distribution form the CI.

Note on pairing level: this is a seed-level paired bootstrap, not a
protein-level one. The protein-level bootstrap (src/protea_reranker_lab/bootstrap.py)
requires the raw prediction parquets, which are gitignored. The
seed-level bootstrap correctly characterises seed-to-seed variability
in the champion lift. See the LB.2 multi-seed sweep section for the
broader variance characterisation.

### Per-cell x aspect paired CI (seed-level, 95%)

Machine-readable form: `experiments/lb3/per_cell_paired_ci.csv`.
Regenerable via `python scripts/lb3_paired_ci.py --write`.

```
cell       champ   [ci_lo,  ci_hi]   base    [ci_lo,  ci_hi]   delta   [ci_lo,  ci_hi]  sig
nk-bpo    0.5596  [0.5571, 0.5618]  0.5333  [0.5333, 0.5333]  0.0263  [0.0238, 0.0285]  *
nk-mfo    0.7065  [0.7041, 0.7112]  0.6447  [0.6447, 0.6447]  0.0618  [0.0594, 0.0665]  *
nk-cco    0.7774  [0.7733, 0.7830]  0.7000  [0.7000, 0.7000]  0.0774  [0.0733, 0.0830]  *
lk-bpo    0.6459  [0.6421, 0.6485]  0.5844  [0.5844, 0.5844]  0.0615  [0.0577, 0.0641]  *
lk-mfo    0.6807  [0.6757, 0.6877]  0.5816  [0.5816, 0.5816]  0.0991  [0.0941, 0.1061]  *
lk-cco    0.7368  [0.7252, 0.7434]  0.7053  [0.7053, 0.7053]  0.0315  [0.0199, 0.0381]  *
pk-bpo    0.4030  [0.4030, 0.4030]  0.4030  [0.4030, 0.4030]  0.0000  [0.0000, 0.0000]
pk-mfo    0.4830  [0.4830, 0.4830]  0.4830  [0.4830, 0.4830]  0.0000  [0.0000, 0.0000]
pk-cco    0.6010  [0.6010, 0.6010]  0.6010  [0.6010, 0.6010]  0.0000  [0.0000, 0.0000]
```

(*) sig_95=1: paired_diff_ci_lo > 0 at the 95% level.

### Reading

All 6 NK+LK cells show statistically significant lifts (CI lower bound
strictly above zero). Smallest lift: lk-cco (+0.0315, CI [+0.0199,
+0.0381]). Largest lift: lk-mfo (+0.0991, CI [+0.0941, +0.1061]).
PK cells are not significant by policy (the reranker is not deployed
on PK; the zero delta is by construction, not a null result).

The CI bounds confirm that the champion bar 0.6215 (selective avg
cafaeval Fmax, 9-cell, 3-seed mean from the LB.2 sweep) is supported
by statistically significant per-cell lifts across all deployed NK+LK
cells. No CI crosses zero. The champion claim is not contradicted.

### Acceptance map

- "Each cell x aspect reported as paired bootstrap CI vs baseline":
  done in the table above and `experiments/lb3/per_cell_paired_ci.csv`.
- "CSV + plot in lab outputs; consumed by thesis chapter 6":
  CSV committed at `experiments/lb3/per_cell_paired_ci.csv`. Plot
  generation is deferred to LM.1 (champion tracking system) which will
  emit dot-and-whisker plots from the committed CSV.

### References

- LB.2 multi-seed sweep (data source, committed 2026-05-17).
- LR.1 closure (structural precedent for this section format).
- `scripts/lb3_paired_ci.py` (regenerator; mirrors `scripts/lr1_lineage_delta.py` layout).
- `experiments/lb3/per_cell_paired_ci.csv` (canonical CSV artefact).
- `tests/test_lb3_paired_ci.py` (regression guard on champion bar).


## LR.1 closure (2026-05-18)

**Status:** done. The v22-architectural lineage booster trained on
`bench-v1-K5-v226-lineage` (LB.1 dataset, id
`3517bc8b-4562-49e0-8c67-99afc5fdc67f`, 24.35M train rows over 13
snapshot pairs, eval v226-v230 over 1.07M rows) is registered in
PROTEA as the nine `v226full_lineage_<cell>` `RerankerModel` rows.

LR.1 maps the plan terminology onto the lab artefacts as follows: the
"v22 booster with lineage feature" is the lambdarank booster trained
with the `v6+lineage-leakfree` bundle (numeric KNN, alignment,
length, taxonomy, go_context and lineage families, with anc2vec and
emb_pca families dropped to remove the historical leakage). That
configuration is the v23 leakage-fixed bundle materialised on the
v226-lineage dataset; the booster artefacts were uploaded on
2026-05-14 with `external_source=protea-reranker-lab@28d9ce0-study_v23`
via `POST /v1/reranker-models/import-by-reference` (the lab uploads
`model.txt` to MinIO directly and PROTEA registers the URI without
re-reading the blob).

### Registered booster rows

```
RerankerModel.id                        name                      cell    artifact_uri
3e5fac6e-f761-473c-9547-041bf8b69c83    v226full_lineage_lk-bpo   lk-bpo  s3://protea/rerankers/.../model.txt
85ef4229-8134-4617-9771-86a584bb66f8    v226full_lineage_lk-cco   lk-cco  s3://protea/rerankers/.../model.txt
5ebb089f-6d7e-4698-adbe-41811fb24744    v226full_lineage_lk-mfo   lk-mfo  s3://protea/rerankers/.../model.txt
dda7948c-551c-49d7-8ac9-87f665e0d79f    v226full_lineage_nk-bpo   nk-bpo  s3://protea/rerankers/.../model.txt
825ed241-2a8a-4f40-b2c9-a9f8f9ec8dc7    v226full_lineage_nk-cco   nk-cco  s3://protea/rerankers/.../model.txt
96e4d02d-d145-4592-8c9d-bf4e58895d01    v226full_lineage_nk-mfo   nk-mfo  s3://protea/rerankers/.../model.txt
4b30b327-f220-4cf9-ba2b-ca6fe58f57ff    v226full_lineage_pk-bpo   pk-bpo  s3://protea/rerankers/.../model.txt
0158529f-45be-421a-b2d6-1fa869a4661d    v226full_lineage_pk-cco   pk-cco  s3://protea/rerankers/.../model.txt
8a4b003f-4ad4-41cb-a3b4-910925a7f8cd    v226full_lineage_pk-mfo   pk-mfo  s3://protea/rerankers/.../model.txt
```

All nine rows carry `dataset_id=3517bc8b-4562-49e0-8c67-99afc5fdc67f`
(the LB.1 dataset row) so the lab-to-PROTEA lineage is recoverable
from the registry.

### Per-cell x aspect lab_fmax delta vs no-lineage baseline (seed 42)

Generated by `python scripts/lr1_lineage_delta.py --write`. The
reranker arm is the v22-architectural booster (lineage features
kept); the baseline arm is the same configuration with the four
`lineage_*` features additionally dropped. Both arms train on
`bench-v1-K5-v226-lineage` at seed 42, num_boost_round=10000,
early_stopping_rounds=100; the delta therefore isolates the
contribution of the lineage feature family on lab Fmax.

```
cell      reranker    baseline      delta
nk-bpo      0.0401      0.0401     0.0000
nk-mfo      0.1927      0.1927     0.0000
nk-cco      0.1473      0.1473     0.0000
lk-bpo      0.0482      0.0498    -0.0016
lk-mfo      0.1526      0.1663    -0.0138
lk-cco      0.1140      0.1140     0.0000
pk-bpo      0.0395      0.0165     0.0230
pk-mfo      0.1374      0.0791     0.0583
pk-cco      0.1191      0.0796     0.0395
```

Machine-readable form: `experiments/lr1/lineage_delta.csv` (committed
record, regenerable from `runs/study_v23` + `runs/study_v24_no_lineage`
artefacts). The CSV is the canonical LR.1 acceptance artefact for
"lab Fmax delta vs baseline reported per cell x aspect".

Reading: lineage features are neutral on NK at seed 42 (early stopping
selects a tree count that does not benefit from the four lineage
columns; see LB.2 multi-seed sweep for the variance characterisation
on NK+LK), neutral-to-slightly-negative on LK (lineage features
crowd out the dominant `neighbor_vote_fraction` signal on the LK lk-mfo
cell), and strongly positive on PK (where lineage features dominate
because PK queries share parent terms with the votes by construction;
this is also the documented DAG-closure shortcut behind the FARM-EXP.10
selective-deploy policy that drops the reranker on PK).

### Canonical cafaeval delta

The lab_fmax delta above is not directly comparable to the
publishable cafaeval Fmax. The LB.2 multi-seed sweep section below
records the canonical cafaeval delta on NK+LK (six cells, three
seeds): the reranker lifts the mean cafaeval Fmax from baseline
0.6249 to 0.6845 (+0.0596). PK is reported separately and the
deployed policy is selective.

### Acceptance map

- "v22 booster trained on bench-v1-K5-v226-lineage": done as
  study_v23 = v22-architectural booster on the v226-lineage dataset
  (LB.1). The nine `runs/study_v23/bench-v1-K5-v226-lineage_<cell>`
  run.json records hold the per-cell `test_fmax` and best iteration.
- "Lab Fmax delta vs baseline reported per cell x aspect": done as
  the table above and `experiments/lr1/lineage_delta.csv`.
- "Booster registered in PROTEA via
  `/v1/reranker-models/import-by-reference`": done; nine rows above,
  all carrying the LB.1 dataset_id and the lab git-sha provenance.

### References

- LB.1 dataset row (PROTEA Postgres) and manifest at
  `datasets/bench-v1-K5-v226-lineage/manifest.json`
  (schema_sha `6d97a624b8a7`).
- LB.2 multi-seed sweep section below (cafaeval delta on NK+LK).
- FARM-EXP.10 champion section below (selective-deploy policy
  including the PK fallback rationale).
- `scripts/lr1_lineage_delta.py` (regenerator for the per-cell delta).


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
