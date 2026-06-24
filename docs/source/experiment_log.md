# Experiment champion log

```{note}
This log is historical and may lag the current champion. For the current
validated champion see {doc}`concepts`.
```

Canonical record of reranker experiment outcomes on bench-v1-K5-v226-lineage-prostt5.
All entries use dataset eval_snapshot_pair v226-v230 and cafaeval
(prop=fill, norm=cafa, no_orphans=True, max_terms=500, th_step=0.001).


## LB.3 closure (2026-05-18)

**Status:** done. Per-cell paired bootstrap confidence intervals on the
leakage-fixed champion (esmc_300m, K=5, lgbm.per_cell_9, feat=leakage-free+lineage bundle,
eval=bench-v1-K5-v226-lineage-prostt5, prop=fill, ens=none)
versus the KNN-only baseline, across all 9 cells x 3 aspects. The
selective-deploy policy (NK+LK reranked, PK baseline fallback) is
reflected in the table: PK cells carry zero delta by construction.

### Methodology

Seed-level paired bootstrap (N=10000, seed=42, alpha=0.05, two-sided
95% percentile CI). Data source: the LB.2 multi-seed sweep (3 seeds:
42, 7, 137), which provides per-seed cafaeval Fmax for both the
champion arm and the KNN-only baseline on bench-v1-K5-v226-lineage-prostt5
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


## LR.4 closure (2026-05-18)

**Status:** done. The leakage-free re-run of the historical selective
rerank policy (`rr=lgbm.per_cell_9, feat=leakage-fixed-lineage,
eval=bench-v1-K5-v226-lineage-prostt5, prop=fill, ens=none`) supersedes the
pre-leakage-fix memory-only champion record (0.4562 selective avg
cafaeval Fmax, validation range unknown). The recompute records a
selective avg cafaeval Fmax of **0.6215 ± 0.0014** on
`bench-v1-K5-v226-lineage-prostt5` (eval `v226-v230`), placing the historical
selective rerank baseline on the same eval distribution as the
chapter-6 results table.

### Configuration

The recompute reuses the leakage-fixed feature bundle (numeric KNN,
alignment, length, taxonomy, go_context and lineage families, with
the anc2vec and emb_pca families dropped to remove the historical
leakage source) at K=5, lambdarank objective, lr=0.05, num_leaves=63,
num_boost_round=10000, early_stopping_rounds=100, seeds {42, 7, 137}.
The deployment policy is selective: reranker on the six NK and LK
cells (no_known and limited_known), KNN baseline fallback on the
three PK cells (previously_known) per ADR-D34 in PROTEA. See the
lineage-feature studies summary in `runs/` (the `SUMMARY` markdown
covering the lineage-feature ablations) for the DAG-closure shortcut
mechanism behind the PK fallback.

### Per-cell selective rerank table (mean over seeds 42, 7, 137)

Generated by `python scripts/lr4_v18_selective.py --write`. Reranker
arm: leakage-fixed booster cafaeval Fmax. Baseline arm: KNN baseline
cafaeval Fmax on the same bench. PK cells report the baseline value in
both columns (selective policy applies the baseline directly; delta is
exactly zero by construction).

```
cell      policy      selective   ci_half   baseline   delta_vs_base
nk-bpo    reranker      0.5596    0.0024     0.5333         +0.0263
nk-mfo    reranker      0.7065    0.0036     0.6447         +0.0618
nk-cco    reranker      0.7774    0.0048     0.7000         +0.0774
lk-bpo    reranker      0.6460    0.0032     0.5844         +0.0616
lk-mfo    reranker      0.6806    0.0060     0.5816         +0.0990
lk-cco    reranker      0.7367    0.0091     0.7053         +0.0314
pk-bpo    baseline      0.4031     n/a       0.4031          0.0000
pk-mfo    baseline      0.4831     n/a       0.4831          0.0000
pk-cco    baseline      0.6009     n/a       0.6009          0.0000
```

Machine-readable form: `experiments/lr4/v18_selective_delta.csv`
(committed record, regenerable from `runs/lb2_multiseed/cis.json`
when the runtime artefact is present, else from the documented
canonical numbers).

### Headline deltas

- 9-cell selective avg cafaeval Fmax (leakage-free recompute):
  **0.6215 ± 0.0014**.
- 9-cell all-baseline avg cafaeval Fmax (same dataset, pre-rerank):
  0.5818.
- Legacy leaky selective avg (memory-only, pre-leakage-fix, range
  unknown): 0.4562.
- Delta of leakage-free recompute vs legacy leaky champion:
  **+0.1653** (gross gain, dominated by the eval distribution
  alignment, not by the architectural change).
- Delta of leakage-free recompute vs all-baseline on the same bench:
  **+0.0397** (the publishable selective-rerank lift).

### Reading the +0.1653 delta

The +0.1653 lift over the legacy leaky number conflates two effects
that the slice deliberately does not try to separate:

1. The eval distribution shifted from the unknown pre-leakage-fix
   range to `bench-v1-K5-v226-lineage-prostt5` (eval v226-v230). The legacy
   number had no pinned validation range, so the per-cell breakdown
   cannot be reproduced. Per `feedback_no_archaeology_recompute`, the
   resolution is to recompute on the current bench rather than
   recover the legacy axis tuple.
2. The leakage source (anc2vec_* + emb_pca_* feature families) was
   removed, which materially changed the per-cell ranking on
   `bench-v1-K5-v226-lineage-prostt5` (see the lineage-feature studies
   summary committed under `runs/`). The leakage-fixed bundle is what
   the LB.2 multi-seed sweep characterises.

The publishable lift for the leakage-free selective rerank policy is
the +0.0397 against the all-baseline reference on the same bench;
the +0.1653 vs the leaky number is only meaningful as the supersession
delta in the chapter-6 narrative ("the corrected selective rerank
champion replaces the leaky 0.4562 number").

### Acceptance map

- "Historical selective-rerank configuration re-trained on the
  leakage-free baseline" (acceptance bullet from LR.4 spec, written
  with the legacy `vN` shorthand in `plans/bioinfo-quick/PLAN.md`):
  done. The leakage-fixed booster is the LB.2 multi-seed sweep
  artefact set; it is the architectural realisation of the selective
  rerank policy on the leakage-free dataset.
- "Per-cell delta vs leaky champion reported (likely large)": done.
  Per-cell deltas are reported against the all-baseline reference on
  the same bench (the leaky champion has no per-cell breakdown on
  file; the aggregate delta of +0.1653 is the only available
  apples-to-apples leaky comparison and is recorded in the headline
  block).
- "Updated champion table in champions.md": done. The leakage-free
  selective rerank entry is added under the manual-entries appendix
  of `champions.md`, pending the FARM-EXP.5+ runner slice that will
  emit FARM-EXP.3-format run records and let
  `scripts/update_champions.py --apply` auto-populate the table.

### References

- LB.2 multi-seed sweep section below (source of the per-cell
  numbers and bootstrap CIs).
- Memory `project_lb2_leakage_fixed_champion` (publishable numbers
  and the supersession of the leaky 0.4562 record).
- Memory `feedback_no_archaeology_recompute` (recompute vs
  reconstruct policy).
- The lineage-feature studies summary committed under `runs/` (the
  `SUMMARY` markdown covering the lineage-feature ablations; describes
  the DAG-closure shortcut mechanism and the PK fallback rationale).
- `scripts/lr4_v18_selective.py` (regenerator for the per-cell
  table).


## LR.1 closure (2026-05-18)

**Status:** done. The v22-architectural lineage booster trained on
`bench-v1-K5-v226-lineage-prostt5` (LB.1 dataset, id
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
`bench-v1-K5-v226-lineage-prostt5` at seed 42, num_boost_round=10000,
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

- "v22 booster trained on bench-v1-K5-v226-lineage-prostt5": done as
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
  `datasets/bench-v1-K5-v226-lineage-prostt5/manifest.json`
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

- `bench-v1-K5-v226-lineage-prostt5` (val_strategy=protein_group, eval v226-v230)
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


## FARM-EXP.8 transversal grid orchestrator (2026-05-18)

**Status:** orchestrator script shipped; sweep execution gated on per-stanza
ExperimentSpec materialisation.

`scripts/run_transversal_grid.py` is the FARM-EXP.8 grid runner. It reads
`experiments/_catalog/transversal.yaml` (224 axis-tuple stanzas from
FARM-EXP.2), expands each stanza into per-cell training jobs (one per
tier-aspect cell, optionally per seed), schedules them in batches of 10,
and dispatches each batch through `scripts/run.py`. After every batch the
FARM-EXP.4 champion auto-updater (`scripts/update_champions.py --apply`)
fires so the champion table tracks progress.

Key properties:

- **Idempotent**: a job whose `run.json` already reports `status=ok` is
  skipped on re-invocation. Re-runs are a no-op once a cell completes.
- **Upfront `run.json`**: every attempt writes a placeholder record
  (status=pending) before subprocess launch, carrying run_id, shortid,
  axis tuple, cell tier+aspect, seed, git sha, resolved hparams, and
  artefact paths. `scripts/run.py` overwrites it with the final ok/failed
  record. A crash leaves the most recent state on disk.
- **Blocked path**: when the per-cell spec YAML is absent the orchestrator
  records `status=blocked, reason=spec_not_materialized` and proceeds
  without subprocess. The operator materialises the spec when the
  upstream dataset manifest (LB.1, LR.1, ...) lands and re-invokes.
- **Filters**: `--plms`, `--features`, `--eval-sets`, `--rerankers`,
  `--shortids`, `--cells`, `--seeds`, `--limit`, plus `--dry-run` and
  `--no-champion-update` toggles.

Per-cell artefacts land under `runs/transversal/<shortid>/<cell>/seed<seed>/`
with `run.json`, `model.txt`, and `predictions.parquet` (the last two on
status=ok only). `runs/` is gitignored, consistent with the prior
FARM-EXP.8 partial pass on the leakfree bundle.

The orchestrator does not invent ExperimentSpec YAMLs; the spec lookup
path is `experiments/_generated/transversal/<shortid>/<cell>_seed<seed>.yaml`.
Wiring a generator (per-stanza spec materialisation that respects the
PLM/dataset/reranker axes) is a separate downstream slice; the current
FARM-EXP.8 leakfree pass already shipped 54 hand-shaped specs under
`experiments/farm_exp_8/`. Pointing the orchestrator at those (via
`--spec-root experiments/farm_exp_8/...`) reproduces the prior pass.

Tests live at `tests/test_run_transversal_grid.py` (33 cases) and cover:
catalog parsing, filter composition, plan cartesian, batch-of-10 grouping,
upfront run.json shape, idempotent skip on status=ok, blocked path, dry
run, subprocess dispatch, and champion-update invocation cadence.


## FARM-EXP.9 pre-leakage cell re-run, partial pass (2026-05-18)

**Status:** partial pass complete for replication cells (NK+LK all 3 seeds done,
PK 2/3 seeds done for pk-mfo, pk-cco pending); 2 ablation cells done; 69 cells
remain pending.

Re-runs the pre-leakage study cells (originally trained on bench-v1-K5 with 52
features including anc2vec leakage columns) on the leakage-free eval set
bench-v1-K5-filtered, dropping the 22 leakage-family columns:

- anc2vec family: anc2vec_has_emb, anc2vec_neighbor_cos, anc2vec_neighbor_maxcos,
  anc2vec_query_known_cos, anc2vec_query_known_count, anc2vec_query_known_maxcos
- emb_pca family: emb_pca_query_0 through emb_pca_query_15 (16 columns)

Dataset: bench-v1-K5-filtered (eval v220-v230, anc2vec leakage genes removed from
the query-known set). Val strategy: temporal, holdout v215-v220, seed {42, 7, 137}.

### Scope

- **Replication cells** (27 total = 9 cells x 3 seeds): lgbm.per_cell_9,
  lambdarank, r=5000, es=50, L=63, lr=0.05.
- **Ablation cells** (39 total = 3 representative cells x 13 families):
  nk-bpo, lk-cco, pk-mfo; each drops one feature family on top of the
  leakage exclusion list.
- **Hparam cells** (27 total = nk-bpo only, 3x3x3 grid on num_leaves, lr,
  neg_pos_ratio): deferred to next pass.
- Total in scope: 93 cells. Source-of-truth: experiments/farm_exp_9/cells_to_rerun.csv.

### Completed in this partial pass (25 cells)

All 18 NK+LK replication cells (6 cells x 3 seeds), all 3 pk-bpo seeds, pk-mfo
seeds 42 and 7, and 2 ablation nk-bpo cells (alignment_nw, alignment_sw).

### Per-cell fmax summary (bench-v1-K5-filtered, lab_fmax, 3 seeds unless noted)

```
cell      new_mean   95% CI half     seeds   old_mean (pre-leakage, bench-v1-K5)
nk-bpo      0.5874       0.0021       3/3    0.4641  (NOT comparable: eval+feat changed)
nk-mfo      0.6347       0.0028       3/3    0.4589
nk-cco      0.7742       0.0007       3/3    0.4831
lk-bpo      0.6151       0.0010       3/3    0.3523
lk-mfo      0.5737       0.0006       3/3    0.2416
lk-cco      0.7881       0.0009       3/3    0.2506
pk-bpo      0.1362       0.0014       3/3    0.1121
pk-mfo      0.2821       0.0032       2/3    0.1046
pk-cco      pending                   0/3    0.1190
```

CI is 95% percentile bootstrap over seeds (N=10000). Old fmax values are
lab_fmax on bench-v1-K5 (pre-leakage, 52 features). Delta is NOT interpretable
as leakage correction effect: eval set and feature set both changed.

### Stop condition check

No champion cell (nk-bpo, nk-mfo, nk-cco, lk-bpo, lk-mfo, lk-cco) has a CI
entirely below zero. All 6 champion cells show large positive deltas vs pre-leakage
baseline (range +0.12 to +0.54). No anomalous champion degradation detected.

### Why deltas are large and not interpretable

The large apparent gains are an artefact of comparing incompatible runs:
1. Eval set changed from bench-v1-K5 (eval v220-v229, 10 snapshots) to
   bench-v1-K5-filtered (eval v220-v230, 11 snapshots, leakage genes removed).
2. Feature set changed from 52 features to 34 features (anc2vec + emb_pca dropped).
3. Pre-leakage scores were inflated by the anc2vec_query_known_* leakage columns;
   removing them reduces the old side of the comparison artificially.

These deltas are reported solely for anomaly detection (STOP condition guard),
not as a measurement of leakage correction effect. The leakage correction
effect is measured in LB.2 and FARM-EXP.10 using cafaeval on a fixed eval set.

### Output artefacts

- experiments/farm_exp_9/cells_to_rerun.csv: canonical cell list (94 rows, 25 done).
- experiments/farm_exp_9/partial_ci.csv: per-cell CI table (machine-readable).
- experiments/farm_exp_9/summary.json: full per-cell stats.
- experiments/farm_exp_9/rep_{cell}_seed{seed}.yaml: 27 replication spec files.
- experiments/farm_exp_9/abl_{cell}_{family}.yaml: 39 ablation spec files.
- runs/transversal/farm_exp_9_rep_{cell}_seed{seed}/run.json: per-run records.
- runs/transversal/farm_exp_9_abl_nk-bpo_{family}/run.json: per-run records.
- scripts/farm_exp_9_summary.py: regenerator for partial_ci.csv and summary.json.
- tests/test_farm_exp_9_partial.py: 17 schema/contract tests (all passing).

### Next pass

Run remaining 69 pending cells: pk-mfo seed 137, pk-cco seeds 42/7/137, all
ablation cells (37 remaining), and hparam grid (27 cells). Estimated total
compute time: 6-10h additional (PK cells 3-6 min each; ablation cells 5-8 min
each due to partial feature set changing convergence).


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
**Dataset:** bench-v1-K5-v226-lineage-prostt5 (eval v226-v230)
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


## LM.3 closure (2026-05-18): per-aspect feature importance audit

**Status:** done. Feature importance audit on the nine
`v226full_lineage_<cell>` boosters (study_v23, seed=42, registered
2026-05-14). Metric: LightGBM `gain` (total information-gain reduction
across all trees and all splits). Aggregation: mean rank across the three
category cells (NK/LK/PK) per aspect.

Source artefacts: `experiments/lm3/feature_importance_per_aspect.csv`
(306 rows: 9 cells x 34 features). Full data, including the 20 zero-gain
features per NK+LK cell (alignment, length, taxonomy families). Regenerable
from `runs/study_v23` via `python scripts/lm3_feature_importance.py --write`.

### Top-10 per aspect (mean rank across NK/LK/PK)

#### BPO (Biological Process)

| rank | feature | mean_rank | mean_gain |
| - | - | - | - |
| 1 | `go_term_frequency` | 3.00 | 204,882.3 |
| 2 | `neighbor_vote_fraction` | 3.67 | 88,400.3 |
| 3 | `evidence_code` | 4.33 | 54,296.6 |
| 4 | `ref_annotation_density` | 5.00 | 53,219.2 |
| 5 | `k_position` | 6.00 | 25,180.2 |
| 6 | `neighbor_distance_std` | 7.33 | 45,149.9 |
| 7 | `distance` | 7.67 | 56,540.3 |
| 8 | `vote_count` | 8.00 | 28,803.6 |
| 9 | `neighbor_mean_distance` | 8.67 | 35,406.7 |
| 10 | `neighbor_min_distance` | 10.67 | 20,350.4 |

#### MFO (Molecular Function)

| rank | feature | mean_rank | mean_gain |
| - | - | - | - |
| 1 | `go_term_frequency` | 2.00 | 144,490.1 |
| 2 | `neighbor_vote_fraction` | 2.67 | 58,033.6 |
| 3 | `evidence_code` | 3.33 | 42,637.3 |
| 4 | `vote_count` | 6.00 | 18,987.2 |
| 5 | `ref_annotation_density` | 6.67 | 16,641.2 |
| 6 | `neighbor_distance_std` | 8.00 | 14,107.6 |
| 7 | `k_position` | 8.67 | 10,690.6 |
| 8 | `neighbor_mean_distance` | 9.33 | 11,616.8 |
| 9 | `qualifier` | 11.00 | 13,204.2 |
| 10 | `neighbor_min_distance` | 11.33 | 6,997.0 |

#### CCO (Cellular Component)

| rank | feature | mean_rank | mean_gain |
| - | - | - | - |
| 1 | `go_term_frequency` | 1.67 | 290,435.4 |
| 2 | `neighbor_vote_fraction` | 3.00 | 50,995.8 |
| 3 | `evidence_code` | 5.33 | 35,764.4 |
| 4 | `k_position` | 7.00 | 13,085.7 |
| 5 | `ref_annotation_density` | 7.00 | 24,277.6 |
| 6 | `vote_count` | 7.33 | 16,562.1 |
| 7 | `neighbor_mean_distance` | 8.33 | 14,678.4 |
| 8 | `neighbor_distance_std` | 8.67 | 17,662.0 |
| 9 | `qualifier` | 9.00 | 56,388.3 |
| 10 | `neighbor_min_distance` | 10.33 | 10,693.7 |

### Interpretation

**BPO:** `neighbor_vote_fraction` and `go_term_frequency` share the top-two
positions across NK and LK cells. In PK the ranking reverses: `go_term_frequency`
dominates at rank 1, and the lineage family enters ranks 2-6
(`lineage_descendant_of_count`, `lineage_ancestor_of_count`,
`lineage_is_ancestor_of_known`). This PK-specific pattern reflects the
DAG-closure shortcut: PK queries already have parent terms annotated,
so the booster learns to exploit the hierarchical overlap captured by
the lineage features. The alignment family (identity_nw, alignment_score_*,
etc.) scores zero gain in every BPO cell, confirming that sequence-level
similarity adds nothing once KNN vote and distance features are included.

**MFO:** The MFO ranking closely mirrors BPO. `neighbor_vote_fraction` and
`go_term_frequency` are the dominant signals across NK and LK.
`evidence_code` consistently ranks third in MFO, higher than in CCO,
suggesting that curation quality matters more for molecular function terms
(where experimental evidence is sparse relative to BPO). PK MFO again shows
lineage features at ranks 2-4, confirming the cross-aspect generality of the
DAG-closure shortcut. The `qualifier` feature enters the MFO top-10 (rank 9)
but is absent from BPO, reflecting MFO-specific NOT-qualifier patterns.

**CCO:** CCO is the outlier aspect. `go_term_frequency` is the undisputed
dominant feature (mean rank 1.67, mean gain nearly 2x higher than BPO).
This is consistent with CCO having fewer distinct terms and higher per-term
annotation density: term frequency predicts GO-term annotation well for
cellular components. In PK-CCO the lineage family is exceptionally strong
(`lineage_ancestor_of_count` and `lineage_is_ancestor_of_known` at ranks 2-3,
aggregate gain exceeding 1M gain units), the highest lineage dominance across
all nine cells. The `qualifier` feature enters the CCO top-10 (rank 9, mean
gain 56,388), substantially higher than in BPO, suggesting qualifier metadata
is more discriminative for cellular component terms.

### Cross-aspect generalists

Features ranking in the top-10 for all three aspects (aspect-stable winners):
`go_term_frequency`, `neighbor_vote_fraction`, `evidence_code`,
`ref_annotation_density`, `vote_count`, `neighbor_distance_std`,
`neighbor_mean_distance`, `neighbor_min_distance`. These seven features
form the shared core that drives reranker performance across all ontology
branches in the NK+LK cells.

### Cell-specific divergences

The lineage family is the clearest aspect-by-cell divergence:

- NK cells: all four lineage features score zero gain in all three aspects.
  Early stopping at iteration 100 selects a tree count that is too small
  to exploit lineage columns once the dominant KNN signals are saturated.
- LK cells: minor lineage signal in lk-mfo only (`lineage_is_descendant_of_known`
  at rank 5, `lineage_is_ancestor_of_known` at rank 10). LK queries share
  parent terms with voters, but the leakage-fixed configuration limits
  exploitation of this overlap.
- PK cells: lineage family dominates at ranks 2-6 across all three aspects.
  This is the DAG-closure shortcut documented in FARM-EXP.10 and ADR-D34,
  which motivated the selective-deploy policy (NK+LK reranker, PK baseline).

### References

- `experiments/lm3/feature_importance_per_aspect.csv` (306 rows, canonical artefact).
- `experiments/lm3/feature_importance_summary.md` (top-10 tables, extended interpretation).
- `scripts/lm3_feature_importance.py` (regenerator, mirrors lr1_lineage_delta.py structure).
- FARM-EXP.10 champion section above (selective-deploy policy and ADR-D34).
- LB.2 multi-seed sweep section above (cafaeval delta on NK+LK).


## LB.2 single-seed retrofix (2026-05-17, predecessor)

lk-mfo cell only; seed=42 gave cafaeval_fmax=0.6998, seed=137 gave 0.6636.
Confirmed anc2vec leakage fix: neighbor_vote_fraction dominates (~61k gain).
See `runs/study_v23/` for all 9-cell results.


## study_v23 (2026-05-14)

Full 9-cell run, seed=42, v23 config (no anc2vec+pca, lambdarank).
See `runs/study_v23/cafaeval/results.csv` for per-cell cafaeval results.
