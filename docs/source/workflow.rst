The lab workflow
================

This page walks the full offline pipeline end to end, from a frozen
PROTEA export to a published booster. Each stage is a small, testable
module; the :func:`~protea_reranker_lab.runner.run_experiment` and
:func:`~protea_reranker_lab.universal_runner.run_universal` entry points
wire them together.

.. contents:: On this page
   :depth: 2
   :local:

Pipeline at a glance
--------------------

.. code-block:: text

   export_research_dataset (PROTEA)
       train.parquet / eval.parquet / manifest.json
                 |
                 v
   1. staging  ──────────  bucket-sorted parquet + labels.npy
   2. features ──────────  frozen 56-feature schema (protea-contracts)
   3. training ──────────  LambdaMART / binary booster (model.txt)
   4. calibration ───────  per-aspect isotonic / Platt map
   5. evaluation ────────  cafaeval f_micro_w + numpy Fmax + bootstrap CI
                 |
                 v
   POST /reranker-models/import-by-reference (PROTEA registry)

Stage 1: staging
----------------

Staging turns a raw export into LightGBM-ready, group-contiguous
buckets without materialising a DataFrame.

Single manifest
~~~~~~~~~~~~~~~

:mod:`protea_reranker_lab.staging` streams ``train.parquet`` through a
two-pass design:

- **Pass 0** scans the heavy string columns, decides the train/val
  split (:func:`~protea_reranker_lab.staging._decide_split`), and caps
  oversized LambdaRank groups.
- **Pass 1** routes every row into one of ``bucket_count`` parquet
  writers keyed by a hash of ``protein_accession``, so that all rows for
  a protein land in the same bucket and stay contiguous after the final
  sort. Contiguity is what lets LightGBM build ranking groups without a
  global sort of the full table.

Pooled multi-manifest
~~~~~~~~~~~~~~~~~~~~~~

:mod:`protea_reranker_lab.pooled_staging` stages all 24 v226-lineage
manifests at once for the universal booster. Memory stays bounded by
**int-coding** the three heavy string columns
(``protein_accession``, ``snapshot_pair``, ``aspect``) into ``int32`` /
``int8`` codes immediately after each source is read, then freeing the
object arrays. Only per-source boolean masks (one byte per row) survive
into Pass 1. ``plm_id`` and ``k_context`` are injected as integer and
float constants per source, since they are not present in the raw
parquet. A global row budget (60 M rows) applies uniform negative
downsampling when the pooled total exceeds the threshold; positives and
validation rows are never dropped.

The extended **four-tuple group key**
(``snapshot_pair``, ``protein``, ``aspect``, ``plm_id``) keeps each
LambdaRank query scoped to a single source and aspect, so the model
never ranks candidates from different PLMs against one another.

Stage 2: features
-----------------

The 56-feature schema is pinned in ``protea-contracts``
(:data:`~protea_contracts.ALL_FEATURES`). The lab never adds a feature
column directly; a schema change must go through the contracts package
so that PROTEA's producer side stays in lockstep. The
:func:`~protea_reranker_lab.schemas.compute_feature_schema_sha` digest is
written into every ``run.json`` and into the dataset manifest, so a
silent column drift invalidates the cache rather than corrupting a
comparison. The universal booster adds two universal-only axes
(``plm_id`` categorical, ``k_context`` float) injected at staging time.

Stage 3: training
-----------------

:mod:`protea_reranker_lab.reranker` wraps LightGBM training and streaming
inference. Buckets are exposed as :class:`lgb.Sequence` objects through
:mod:`protea_reranker_lab.sequences`, enabling ``free_raw_data=True`` so
peak RSS is capped regardless of dataset size. The default objective is
``lambdarank`` over protein groups; the per-cell champion uses a binary
objective instead (see :doc:`overview`). IA-aware training (sample
weights and a custom ``feval`` callback) lives in
:mod:`protea_reranker_lab.ia_weighting`; see :doc:`ia_aligned_training`.

Stage 4: calibration
--------------------

:mod:`protea_reranker_lab.calibration` fits an optional per-aspect score
map (isotonic regression or Platt scaling) on the held-out validation
band, so that raw booster margins become comparable probabilities before
fusion or thresholding. Calibration is identity when no estimator is
fitted, which keeps the default path unchanged. The optional
:mod:`protea_reranker_lab.hierarchical_correction` enforces the GO DAG
constraint that a parent term scores at least as high as its
highest-scoring child.

Stage 5: evaluation
-------------------

Three evaluators serve different questions:

- :func:`~protea_reranker_lab.evaluate.fmax_per_protein_group` is the
  fast numpy proxy used during training and diagnostics (no propagation,
  no IA weighting).
- :func:`~protea_reranker_lab.evaluate.eval_f_micro_w` is the canonical
  selection metric: it drives the ``cafaeval`` subprocess with the band
  IA table and returns IA-weighted ``f_micro_w`` (see :doc:`metrics`).
- :mod:`protea_reranker_lab.bootstrap` and
  :mod:`protea_reranker_lab.compare` run a per-protein paired bootstrap
  of Fmax against the KNN ``vote_count`` baseline to produce 95 percent
  confidence intervals and a one-sided p-value.

:mod:`protea_reranker_lab.recall` separately reports the KNN retrieval
ceiling so the reranker's ranking lift is never conflated with the
candidate generator's recall limit.

Reproducibility artefacts
-------------------------

Every run writes a ``spec.yaml`` at the start (so a crash leaves a
recoverable trace) and a ``run.json`` at the end capturing resolved
hyperparameters, dataset lineage (spec and schema digests), feature
list, timings, and final metrics. The universal booster additionally
writes ``pooled_staging_meta.json`` (row counts, row budget, peak RSS)
and ``plm_ablation_summary.json``.
