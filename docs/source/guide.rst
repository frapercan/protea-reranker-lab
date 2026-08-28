Stage guide
===========

The :doc:`concepts` page tells the workflow as a story. This page expands
each stage with the real entrypoints and the mechanics that matter when you
read or extend the code. The two orchestrators
:func:`~protea_reranker_lab.runner.run_experiment` (per cell) and
:func:`~protea_reranker_lab.universal_runner.run_universal` (pooled) wire
the stages together.

.. contents:: On this page
   :depth: 2
   :local:

Stage 1: staging
----------------

Staging turns a raw export into LightGBM-ready, group-contiguous buckets
without materialising a DataFrame.

Single manifest
~~~~~~~~~~~~~~~

:func:`protea_reranker_lab.staging.stage_for_training` streams
``train.parquet`` through a two-pass design:

- **Pass 0** scans the heavy string columns, decides the train/val split
  (:func:`~protea_reranker_lab.staging._decide_split`), and caps oversized
  LambdaRank groups.
- **Pass 1** routes every row into one of ``bucket_count`` parquet writers
  keyed by a hash of ``protein_accession``, so that all rows for a protein
  land in the same bucket and stay contiguous after the final sort.
  Contiguity is what lets LightGBM build ranking groups without a global
  sort of the full table.

The stage emits, per split, the sorted bucket parquets plus row-aligned
``labels.npy`` and ``groups.npy`` sidecars; when IA weighting is requested
it also emits ``go_terms.npy`` (see :doc:`ia_aligned_training`).

Pooled multi-manifest
~~~~~~~~~~~~~~~~~~~~~~

:func:`protea_reranker_lab.pooled_staging.stage_for_training_pooled` stages
all 24 v226-lineage manifests at once for the universal booster. Memory
stays bounded by **int-coding** the three heavy string columns
(``protein_accession``, ``snapshot_pair``, ``aspect``) into ``int32`` and
``int8`` codes immediately after each source is read, then freeing the
object arrays. Only per-source boolean masks (one byte per row) survive
into Pass 1. ``plm_id`` and ``k_context`` are injected as integer and float
constants per source, since they are not present in the raw parquet. A
global row budget (60 M rows) applies uniform negative downsampling when
the pooled total exceeds the threshold; positives and validation rows are
never dropped.

The extended **four-tuple group key**
(``snapshot_pair``, ``protein_accession``, ``aspect``, ``plm_id``) keeps
each LambdaRank query scoped to a single source and aspect, so the model
never ranks candidates from different PLMs against one another (see
:ref:`universal-staging`).

Stage 2: features
-----------------

The feature schema is pinned in ``protea-contracts``
(:data:`~protea_contracts.ALL_FEATURES`). The lab never adds a feature
column directly; a schema change must go through the contracts package so
that PROTEA's producer side stays in lockstep. Training does not use that
catalogue wholesale: the default set is
:data:`~protea_reranker_lab.contracts.DEFAULT_TRAINING_FEATURES`, which holds
out the families listed in
:data:`~protea_reranker_lab.contracts.UNADOPTED_FEATURE_FAMILIES` (declared
but unproduced signals, and the ``lineage`` family excluded by the leakage
ruling). Set ``enabled_feature_families`` to train with them. The
:func:`~protea_reranker_lab.schemas.compute_feature_schema_sha` digest is
written into every ``run.json`` and into the dataset manifest, so a silent
column drift invalidates the cache rather than corrupting a comparison.
The runner splits the selected columns into numeric and categorical sets
from the contracts ``NUMERIC_FEATURES`` and ``CATEGORICAL_FEATURES`` lists
before handing them to LightGBM. The universal booster adds two
universal-only axes (``plm_id`` categorical, ``k_context`` float) injected
at staging time.

Stage 3: training
-----------------

:mod:`protea_reranker_lab.reranker` wraps LightGBM training and streaming
inference. Buckets are exposed as :class:`lgb.Sequence` objects through
:mod:`protea_reranker_lab.sequences`, enabling ``free_raw_data=True`` so
peak RSS is capped regardless of dataset size. The default objective is
``lambdarank`` over protein groups; the per-cell champion uses a binary
objective instead (see :doc:`concepts`). IA-aware training (sample weights
and a custom ``feval`` callback) lives in
:mod:`protea_reranker_lab.ia_weighting`; see :doc:`ia_aligned_training`.
The per-cell path runs through
:func:`~protea_reranker_lab.runner.run_experiment`, which fits the booster,
scores the eval split, and writes ``model.txt``, ``predictions.parquet``,
and ``run.json``.

Stage 4: calibration
--------------------

:mod:`protea_reranker_lab.calibration` fits an optional per-aspect score map
(:func:`~protea_reranker_lab.calibration.fit_aspect_calibrators`, isotonic
regression or Platt scaling) on the held-out validation band, so that raw
booster margins become comparable probabilities before fusion or
thresholding. An aspect with fewer than ``min_samples`` rows, or with no
positive label, falls back to identity, so the default path is unchanged.
:func:`~protea_reranker_lab.calibration.calibrate_scores` applies the
fitted maps to a flat score array, and the calibrators plus a JSON sidecar
are persisted with
:func:`~protea_reranker_lab.calibration.save_calibrators`. The optional
:mod:`protea_reranker_lab.hierarchical_correction` then enforces the GO DAG
constraint that a parent term scores at least as high as its
highest-scoring child.

Stage 5: evaluation
-------------------

Three evaluators serve different questions:

- :func:`~protea_reranker_lab.evaluate.fmax_per_protein_group` is the fast
  numpy proxy used during training and diagnostics (no propagation, no IA
  weighting).
- :func:`~protea_reranker_lab.evaluate.eval_f_micro_w` is the canonical
  selection metric: it drives the ``cafaeval`` subprocess with the band IA
  table and returns IA-weighted ``f_micro_w`` (see :doc:`metrics`). The IA
  table and ontology snapshot are resolved through
  :func:`~protea_reranker_lab.band_registry_bridge.resolve_band_artifacts`,
  which refuses to pair a cell with artifacts from a foreign band.
- :mod:`protea_reranker_lab.bootstrap` and
  :mod:`protea_reranker_lab.compare` run a per-protein paired bootstrap of
  Fmax against the KNN ``vote_count`` baseline to produce 95 percent
  confidence intervals and a one-sided p-value.

:mod:`protea_reranker_lab.recall` separately reports the KNN retrieval
ceiling so the reranker's ranking lift is never conflated with the
candidate generator's recall limit.

Reproducibility artefacts
-------------------------

Every run writes a ``spec.yaml`` at the start (so a crash leaves a
recoverable trace) and a ``run.json`` at the end capturing resolved
hyperparameters, dataset lineage (spec and schema digests), feature list,
timings, and final metrics. The universal booster additionally writes
``pooled_staging_meta.json`` (row counts, row budget, peak RSS) and
``plm_ablation_summary.json``.
