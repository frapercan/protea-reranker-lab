# ADR D41: Universal booster pipeline (F-RERANK-UNIVERSAL)

**Status:** Accepted
**Date:** 2026-06-08

## Context

Phase 3a shipped one LightGBM booster per (category, aspect, PLM, K) cell.
The 24-cell grid of separate models made cross-PLM and cross-K comparison
straightforward, but it also duplicated a large fraction of the feature
computation, prevented the model from learning cross-PLM disambiguation, and
produced a collection of artifacts that are awkward to deploy in a single LAFA
submission.

The F-RERANK-UNIVERSAL loop replaces these per-cell models with a single
universal booster that trains over all 24 v226-lineage manifests simultaneously
(8 PLM x K{3,5,10}), is aspect-conditioned (sees all three GO namespaces), and
uses IA-weighted LambdaMART as its training objective. The result is one
booster artifact per protein category (NK or LK) that captures cross-PLM and
cross-K signal that the per-cell approach cannot observe.

## Decisions

### 1. Pooled multi-manifest staging (no physical combined parquet)

The staging layer in `pooled_staging.py` streams each of the 24 source
manifests through shared bucket writers without writing a physical combined
parquet to disk. This avoids the write-OOM failure pattern documented in
PROTEA memory (`feedback_minijob_write_oom_2026_05_28.md`): the minijob write
worker previously pinned all shards in RAM (54 GB OOM on a 62 GB box).

**Memory design (Pass 0 / Pass 1):**

Pass 0 reads one source at a time, int-codes the three heavy string columns
(`protein_accession`, `snapshot_pair`, `aspect`) from Python object arrays into
`int32` / `int8` codes immediately after concatenation, computes the keep/val
masks, then frees the row arrays. Only per-source boolean masks (1 byte/row)
survive into Pass 1. Peak per-source RSS is approximately 500 MB (26 M rows x
10 B/row int-coded) versus 5-20 GB with raw Python-object arrays.

The key function is `_strings_to_int32` / `_strings_to_int8`, which uses
`np.unique` with `return_inverse=True` to vectorise the factorisation. The
intermediate string objects are deleted explicitly with `del` before the next
step to avoid the internal sort in `np.unique` doubling peak RSS on a live
object array.

Pass 1 routes each source through the globally-opened bucket writers using the
pre-computed masks, injecting `plm_id` and `k_context` as integer constants
per source (they are not present in the raw parquet, only in the manifest).

**Global row budget:** `_GLOBAL_ROW_BUDGET = 60_000_000` rows across all
sources. The budget is derived from LightGBM's peak RSS per row at 56 features
(float64 raw + uint8 bins + grad/hess = ~520 B/row) and a 30 GB training
headroom target on a 62 GB box. If the total kept rows exceed the budget, train
negatives are downsampled uniformly across sources at a single
`downsample_factor < 1.0`. Positives and val rows are never dropped. The
factor, peak RSS, and row counts are written to `pooled_staging_meta.json` for
reproducibility.

### 2. Aspect-conditioned single booster

The universal booster is trained on one protein category (e.g. `nk`) but with
all three ontology aspects (MFO, BPO, CCO) in the same training set. The
`aspect` column is carried as a categorical feature so the booster can learn
aspect-specific decision boundaries. The group key for LambdaMART is the
4-tuple `(snapshot_pair, protein_accession, aspect, plm_id)` to prevent
incoherent labels across snapshot deltas from polluting the same ranking group.

### 3. K-augmented feature space

`plm_id` (categorical, codes: 0..7) and `k_context` (float32, values 3/5/10)
are injected per source as features. A single booster therefore sees all
combinations of PLM and K as separate training examples and can learn
cross-PLM and cross-K patterns directly. The `plm_id` ablation run
(`plm_id_ablation=True`) trains an identical booster with `plm_id` dropped and
reports the contribution in `plm_ablation_summary.json`.

### 4. IA-weighted LambdaMART objective

The training objective is LambdaMART (`objective: lambdarank`) with
IA-weighting in `combined` mode: IA weights are applied both as sample weights
(`lgb.Dataset(weight=...)`) and via a custom `feval` callback that computes an
IA-weighted ranking metric at each early-stopping round. The IA table is
`datasets/ia/IA-swissprot-exp-v227.txt` (provenance in ADR D40 and
`docs/source/metrics.rst`).

Label propagation (`parent_map_path`) is NOT supported in the pooled path
because materialising the full protein+GO arrays across all 24 sources defeats
the memory budget. Set `plan.parent_map_path=None` for pooled training.

### 5. Temporal validation protocol

The train/val/test split follows a strict temporal order:

```
train rows    <- snapshot pairs BEFORE val_holdout_snapshot
val rows      <- exactly val_holdout_snapshot (e.g. "v220-v226")
test / eval   <- reserved eval.parquet (v226-v230, written at dataset export time)
```

The `val_holdout_snapshot` mechanism in `StagePlan` matches rows whose
`snapshot_pair` column equals the holdout string and routes them to the val
split. In the int-coded Pass 0 path, the string is resolved to its `int32` code
in the pair vocabulary before calling `_decide_split`, avoiding the unsafe
`if not 0` branch that would raise `ValueError` on code 0.

**Why the v226-v227 interim band was degenerate:** the first PoC used the
v226-v227 delta as the validation set, but this snapshot pair contains only 9
to 16 GT rows per cell (it captures only proteins newly annotated between those
two GOA snapshots, not the full protein universe). cafaeval cannot produce
meaningful `f_micro_w` from single-digit GT row counts and returns `None`.
The holdout band is therefore set to `"v220-v226"` (hundreds to thousands of
GT rows per cell), which sits strictly between training (<v220-v226) and the
reserved test (v226-v230 eval.parquet).

### 6. Holdout-band f_micro_w evaluator

`_eval_holdout_band` in `universal_train.py` scores the held-out `stage.val`
split with both the trained booster and the KNN-only `vote_count` baseline,
then calls cafaeval per NK/LK cell of the trained category. The evaluator runs
before the staging directory is torn down (because `stage.val` bucket parquets
live in the staging dir), and persists results in `run.json` under
`valid_band_metrics`.

The cafaeval parser was fixed in this loop (PR #75): the previous parser keyed
cafaeval's `dfs_best` output by namespace and looked for a non-existent
`"metric"` field, so every `f_micro_w` value was `None`. The corrected parser
keys by metric-KIND (`"f_micro_w"`, `"f_micro"`, `"f"`) and reads the column
of the same name from the best-threshold row whose `ns` matches the aspect
namespace.

The evaluator result note in `run.json` reads: "Held-out snapshot-pair
validation GT (candidate-set restricted). Reranker-vs-KNN comparison is valid;
absolute f_micro_w may be optimistic vs a clean v227-lineage recompute
(deferred)." The absolute number should not be treated as final; it depends
on the candidate set available in the v226-lineage exports.

### 7. GOA self-prior (leakage-safe t0 annotations)

The GOA self-prior is the set of non-experimental (`IEA` and other
computational) GOA annotations for a query protein at t0 (before the test
period begins). These annotations are leakage-safe because they are derived
from the protein's sequence-based and orthology-based annotation at the time of
the LAFA snapshot, not from future experimental evidence. They supplement the
KNN candidate list with terms the embedding-based retrieval might miss, without
introducing temporal leakage.

The self-prior concept lives in PROTEA core
(`apps/method_runtime/self_prior.py`) as a reusable optional candidate source,
validated in slice F-LAFA-SELFPRIOR.1 and intended for inclusion in the next
LAFA submission bundle. The lab references this source but does not re-implement
it: the universal booster trains on feature rows exported by PROTEA, which may
or may not include self-prior candidates depending on the export configuration.

## PoC status and deferred work

The universal booster as of PR #75 is a proof-of-concept validated on the
`"v220-v226"` holdout band. The headline finding is that the universal booster
beats the KNN baseline (`prot_t5 K3`) on the held-out validation band, with a
positive delta on NK+LK mean `f_micro_w`. The absolute number is
candidate-set restricted and should not be cited as the final production result.

Two items are deferred:

1. **Clean v227-lineage recompute.** A full export and training run on the
   v227-lineage datasets (v227-v230 as the reserved test band, v220-v226 as
   the held-out validation, all exports regenerated from the v227 GOA snapshot)
   is required before the universal booster's absolute `f_micro_w` can be
   published alongside the per-cell numbers.

2. **GOA self-prior bundle.** Integrating the self-prior candidate source into
   the LAFA submission bundle with the universal booster is tracked in
   F-LAFA-SELFPRIOR.1 and is not part of this ADR.

## Artifacts produced per run

| File | Description |
|---|---|
| `model.txt` | LightGBM universal booster (text format) |
| `calibrators/` | Per-aspect isotonic calibration objects |
| `predictions.parquet` | Eval split predictions (raw + corrected score) |
| `run.json` | Full lineage: manifest pool, feature list, staging stats, metrics |
| `pooled_staging_meta.json` | Staging row counts, row budget, peak RSS |
| `plm_ablation_summary.json` | plm_id contribution to f_micro_w |

## Consequences

- One booster replaces 24+ per-cell models for NK (or LK) cells.
- The pooled staging path (`stage_for_training_pooled`) is incompatible with
  label propagation; this is intentional and documented.
- The `val_holdout_snapshot` must name a snapshot pair with sufficient GT rows
  (hundreds+); the v226-v227 interim is documented as degenerate.
- The cafaeval parser fix (PR #75) applies to all future uses of `_run_cafaeval`
  in the lab, including the holdout-band evaluator and the legacy VALID-band
  probe.
