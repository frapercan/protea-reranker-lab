Universal reranker (F-RERANK-UNIVERSAL)
=======================================

This page documents the universal booster pipeline introduced in the
F-RERANK-UNIVERSAL loop (PRs #64 to #75). The formal design decisions are
recorded in `ADR D41 <https://github.com/frapercan/protea-reranker-lab/blob/develop/docs/adr/D41-universal-reranker.md>`_.

.. contents:: On this page
   :depth: 2
   :local:

Motivation
----------

Phase 3a shipped one LightGBM booster per (category, aspect, PLM, K) cell
(24+ models in total). Training separate models prevents the booster from
learning cross-PLM disambiguation and cross-K generalization signals. Deploying
24 artifacts in a single LAFA submission is also operationally awkward.

The universal booster replaces these per-cell models with a **single artifact
per protein category** (NK or LK) that trains over all 24 v226-lineage manifests
simultaneously (8 PLM x K{3,5,10}), is aspect-conditioned (all three GO
namespaces in one model), and uses IA-weighted LambdaMART as its training
objective.

Pipeline overview
-----------------

The entry point is :func:`protea_reranker_lab.universal_runner.run_universal`.
The script ``scripts/run_universal_booster.py`` wraps it with a CLI. Training
helpers (staging, fitting, holdout evaluation) live in
:mod:`protea_reranker_lab.universal_train` (internal module, not part of the
public API).

**Stage 1: manifest discovery.** ``_discover_manifests`` scans the
``datasets_root`` for subdirectories matching
``bench-v1-K{k}-v226-lineage-{plm}`` for every (PLM, K) combination in the
pool. Missing manifests are logged but do not abort the run; ``run.json``
records the coverage percentage.

**Stage 2: pooled staging.** :func:`protea_reranker_lab.pooled_staging.stage_for_training_pooled`
streams all 24 sources through shared bucket writers. The full staging memory
design is described in :ref:`universal-staging`.

**Stage 3: LightGBM fit.** The pooled staged splits are fed to
:func:`protea_reranker_lab.reranker.fit` with IA weights and the IA-feval
callback. The booster is saved to ``model.txt``.

**Stage 4: calibration.** Per-aspect isotonic calibration is fit on the
eval split and saved to ``calibrators/``.

**Stage 5: hierarchical correction.** Post-hoc hierarchical-consistency
correction (parent score >= max child score) is applied using ``parent_map.json``
from the shared GO ontology snapshot.

**Stage 6: holdout-band evaluation.** :func:`~protea_reranker_lab.universal_train._eval_holdout_band`
scores the held-out validation split with both the booster and the KNN
``vote_count`` baseline; cafaeval is called per NK/LK cell (see
:ref:`universal-holdout-eval`).

**Stage 7: plm_id ablation (optional).** A second run with ``plm_id`` dropped
is launched as a guard; the contribution of ``plm_id`` to ``f_micro_w`` is
written to ``plm_ablation_summary.json``.

.. _universal-staging:

Pooled multi-manifest staging
------------------------------

Module: :mod:`protea_reranker_lab.pooled_staging`

The pooled staging layer processes 24 parquet sources without writing a
physical combined parquet. It uses a two-pass design.

**Pass 0 (per-source scan and mask computation):**

For each source in turn, the train parquet is streamed and four columns are
read: ``protein_accession``, ``label``, ``snapshot_pair``, ``aspect``. These
are concatenated, then int-coded immediately:

- ``protein_accession`` (high-cardinality strings): ``int32`` via
  ``_strings_to_int32``.
- ``snapshot_pair`` (medium-cardinality, e.g. ``"v220-v226"``): ``int32``.
- ``aspect`` (3 values): ``int8`` via ``_strings_to_int8``.

The raw string arrays are deleted with ``del`` before ``np.unique`` runs on
the next column. This prevents the internal sort in ``np.unique`` from doubling
peak RSS on a live Python object array. After int-coding, keep and val masks
are computed and the int-coded arrays are freed; only the boolean masks (1
byte/row) survive to Pass 1.

Peak per-source RSS is approximately 500 MB for a 26 M-row source. With 24
sources, Pass 0 operates sequentially and the maximum simultaneous footprint
is one source worth of data.

**Global row budget:** If the total kept rows across all sources exceed
``_GLOBAL_ROW_BUDGET`` (60 M rows, derived from a 30 GB LightGBM training
headroom target), train negatives are downsampled uniformly across sources at a
single ``downsample_factor``. Positives and val rows are never dropped. The
factor, peak RSS, and row counts are written to ``pooled_staging_meta.json``.

**Pass 1 (write to bucket writers):** Each source is streamed again; per-row
routing uses the pre-computed masks. ``plm_id`` and ``k_context`` are injected
as integer / float constants per source (they are not present in the raw
parquet). Rows are routed to train or val bucket writers based on the boolean
masks. The eval split is produced from the primary source only (prot_t5 K10
preferred; falls back to any K10 source, then to the first available).

**Extended group key:** The LambdaMART group key is the 4-tuple
``(snapshot_pair, protein_accession, aspect, plm_id)``, carried as
``snapshot_pair`` in bucket reserved columns. This prevents incoherent labels
across snapshot deltas from polluting the same ranking group.

Temporal validation protocol
------------------------------

The train / val / reserved-test split follows a strict temporal order:

.. code-block:: text

   train    <- snapshot pairs BEFORE val_holdout_snapshot
   val      <- exactly val_holdout_snapshot (default: "v220-v226")
   test     <- reserved eval.parquet, never seen during training or validation
               (v226-v230 window, written at dataset export time)

The ``val_holdout_snapshot`` string is set in :class:`~protea_reranker_lab.staging.StagePlan`
and resolved to its ``int32`` code in the pair vocabulary during Pass 0
(before ``_decide_split`` is called), which avoids a false-negative branch
on pair code 0.

**Why ``"v220-v226"`` and not ``"v226-v227"``:** The v226-v227 GOA delta
contains only proteins newly annotated between those two releases, typically
9 to 16 GT rows per cell. cafaeval cannot compute a meaningful ``f_micro_w``
from single-digit GT row counts and returns ``None`` for every metric. The
``"v220-v226"`` band contains the full v220-era protein universe annotated
through v226 and provides hundreds to thousands of GT rows per cell, making it
a valid validation surface. It sits strictly between training (snapshot pairs
before v220-v226) and the reserved test (v226-v230 eval.parquet).

.. _universal-holdout-eval:

Holdout-band evaluator (f_micro_w)
------------------------------------

``_eval_holdout_band`` in :mod:`protea_reranker_lab.universal_train` evaluates
cafaeval on the held-out ``stage.val`` split before the staging directory is
torn down. It:

1. Reads row-aligned arrays from the val split sidecars: ``protein_accession``,
   ``go_term_id``, ``label``, ``aspect`` (one-per-group, expanded by groups),
   and ``vote_count`` for the KNN baseline.
2. Runs the trained booster over the val feature buckets via
   :func:`~protea_reranker_lab.reranker.predict_streaming`.
3. Writes ``pred.tsv`` and ``gt.tsv`` per NK/LK cell in a temporary directory.
4. Calls cafaeval via a subprocess driver for each cell and parses
   ``f_micro_w``, ``f_micro``, and ``fmax``.
5. Repeats steps 3 to 4 with the KNN ``vote_count`` scores for the baseline.
6. Aggregates per-cell results and computes NK+LK mean ``f_micro_w`` for both
   the booster and the KNN baseline.
7. Persists the result in ``run.json`` under ``valid_band_metrics``.

**cafaeval parser fix (PR #75):** The previous parser keyed cafaeval's
``dfs_best`` JSON output by namespace string (e.g. ``"molecular_function"``)
and looked for a non-existent field ``"metric"`` in each row, so every
``f_micro_w`` value was ``None``. The corrected parser keys by metric-KIND
(``"f_micro_w"``, ``"f_micro"``, ``"f"``), reads the column of the same name
from the best-threshold row, and matches rows by the ``ns`` field. This fix
applies to all ``_run_cafaeval`` callers in the lab.

**Interpreting results:** ``valid_band_metrics.reranker_mean_f_micro_w`` vs
``valid_band_metrics.knn_mean_f_micro_w`` is a valid delta comparison because
both arms share the same candidate set. The absolute values are
candidate-set restricted (optimistic vs a full-pipeline CAFA submission; see
:ref:`metrics-recovery-ceiling` in :doc:`metrics`). A clean v227-lineage
recompute is needed before the absolute numbers can be cited in published output.

GOA self-prior
--------------

The GOA self-prior is the set of non-experimental (IEA and other computational)
GOA annotations for a query protein at t0 (the start of the LAFA test period).
These annotations are leakage-safe: they are derived from sequence-based and
orthology-based evidence that exists before the experimental annotations that
form the test ground truth.

**Purpose.** KNN retrieval over PLM embeddings may miss GO terms for which no
similar protein exists in the embedding space. The self-prior supplements the
candidate list with terms from the protein's own computational annotation
record, without introducing temporal leakage (computational annotations do not
depend on the experimental ground truth used for evaluation).

**Where it lives.** The self-prior implementation is in PROTEA core
(``apps/method_runtime/self_prior.py``), validated in slice F-LAFA-SELFPRIOR.1.
The lab does not re-implement the source: the universal booster trains on
feature rows exported by PROTEA, which may include self-prior candidates
depending on the export configuration used. Integration into the LAFA
submission bundle is tracked in F-LAFA-SELFPRIOR.1 and is not part of PR #75.

Running the universal booster
-------------------------------

The CLI entry point is ``scripts/run_universal_booster.py``:

.. code-block:: bash

   poetry run python scripts/run_universal_booster.py \
       --datasets-root /path/to/lab/datasets \
       --out-dir runs/universal_booster \
       --v227-delta-parquet /path/to/v227/train.parquet \
       --category nk \
       --ia-weighting all \
       --seed 42 \
       --plm-id-ablation

Key outputs under ``--out-dir``:

.. code-block:: text

   model.txt                   LightGBM universal booster
   calibrators/                isotonic calibrators per aspect
   predictions.parquet         raw + calibrated + corrected scores
   run.json                    full lineage artifact
   pooled_staging_meta.json    staging row counts, row budget, peak RSS
   plm_ablation_summary.json   plm_id feature contribution

The ``--datasets-root`` must contain subdirectories named
``bench-v1-K{k}-v226-lineage-{plm}`` for each PLM and K in the pool.
Shared GO ontology files (``go.obo``, ``parent_map.json``) are looked up from
``datasets/bench-v1-K5-filtered/`` or ``datasets/bench-v1-K5/`` (the canonical
shared-snapshot copies); a ``--obo`` or ``--parent-map`` override falls back to
these locations if the supplied path does not exist.

The IA table is resolved from ``--ia-path`` or defaults to
``datasets/ia/IA-swissprot-exp-v227.txt`` (provenance: :ref:`metrics-ia-provenance`).

PoC status and deferred work
------------------------------

The universal booster as shipped in PR #75 is a proof-of-concept:

- **Validated:** beats the ``prot_t5 K3`` KNN baseline on NK+LK mean
  ``f_micro_w`` on the ``"v220-v226"`` holdout band.
- **Candidate-set restricted:** absolute ``f_micro_w`` is optimistic relative
  to a full-pipeline CAFA submission (see :ref:`metrics-recovery-ceiling`).
- **Deferred:** a clean v227-lineage recompute (all exports regenerated from
  the v227 GOA snapshot, v227-v230 as the reserved test band) is required
  before the universal booster's absolute metrics can be published.
- **GOA self-prior:** bundling the self-prior candidate source with the next
  LAFA submission is tracked in F-LAFA-SELFPRIOR.1.
