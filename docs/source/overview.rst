Overview
========

**protea-reranker-lab** is a self-contained experimentation environment
for the LightGBM-based GO-term reranker used in PROTEA. It takes the
raw 56-feature parquet dumps produced by PROTEA's
``export_research_dataset`` operation and trains a streaming ranking
model without ever materialising full data frames in memory.

Published champion
------------------

The per-cell champion (binary-objective, multi-seed, 2026-05-18) is a
selective-average cafaeval Fmax of **0.7291 +/- 0.0028** on
``bench-v1-K5-v226-lineage`` (NK+LK cells, 3 seeds). All six NK+LK
paired-bootstrap confidence intervals are strictly positive at 95 %
(N=10000 bootstrap iterations per cell). This figure is cited in
Chapter 6 of the doctoral thesis (selective deployment: NK+LK cells use
the binary-objective champion; PK cells remain on the KNN baseline per
ADR D34).

An earlier number (0.4562) arose from a train-parquet construction
issue where the GO category column was replicated across snapshots for
the same protein-term pair, inflating the effective training signal
without introducing temporal leakage. The fix is tracked in PROTEA
memory key ``project_anc2vec_leakage_mechanism``. Do not cite 0.4562
in any published output. The intermediate LB.2 estimate (0.6215 +/-
0.0014, leakage-fixed multi-seed) is also superseded and should not be
cited in new writing.

Universal booster (F-RERANK-UNIVERSAL, PoC)
--------------------------------------------

The F-RERANK-UNIVERSAL loop (PRs #64 to #75) introduces a **universal
booster**: a single aspect-conditioned, K-augmented, IA-weighted
LambdaMART model trained simultaneously over all 24 v226-lineage
manifests (8 PLM x K{3,5,10}), replacing the per-cell phase3a models.
Key design decisions are recorded in ADR D41
(``docs/adr/D41-universal-reranker.md``). The PoC validates on the
held-out ``v220-v226`` band and beats the ``prot_t5 K3`` KNN baseline
on NK+LK mean ``f_micro_w``. The absolute number is candidate-set
restricted; a clean v227-lineage recompute is pending before the result
can be published alongside the per-cell champion.

Pooled multi-manifest staging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:mod:`protea_reranker_lab.pooled_staging` streams each of the 24 source
manifests through shared bucket writers without writing a physical
combined parquet. Memory is bounded by int-coding the three heavy string
columns (``protein_accession``, ``snapshot_pair``, ``aspect``) from
Python object arrays into ``int32`` / ``int8`` codes immediately after
reading each source, then freeing the raw arrays. Only per-source
boolean masks survive into the write pass (Pass 1). A global row budget
(60 M rows across all sources) applies uniform negative downsampling if
the pooled total exceeds the threshold; positives and val rows are never
dropped.

``plm_id`` and ``k_context`` are injected as integer / float constants
per source at Pass 1 time: they do not appear in the raw parquet and are
not in the physical train split until the bucket write pass.

Temporal validation protocol
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The train/val/test split follows a strict temporal order:

- **train**: all snapshot pairs before ``val_holdout_snapshot``
  (default ``"v220-v226"``)
- **val**: rows whose ``snapshot_pair`` equals ``val_holdout_snapshot``
  exactly (the holdout band, hundreds to thousands of GT rows per cell)
- **eval / test**: the reserved ``eval.parquet`` (v226-v230, written at
  dataset export time and never seen during training or validation)

The ``v226-v227`` interim band that the first PoC used as validation is
documented as degenerate: it contains only 9 to 16 GT rows per cell
(only proteins newly annotated between those two GOA releases), which is
too sparse for cafaeval to produce a meaningful ``f_micro_w``.

Holdout-band f_micro_w evaluator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``_eval_holdout_band`` (in :mod:`protea_reranker_lab.universal_train`)
scores the held-out ``stage.val`` split with both the trained booster
and the KNN-only ``vote_count`` baseline, then calls cafaeval per NK/LK
cell. It runs before the staging directory is torn down (the bucket
parquets are needed). Results are written to ``run.json`` under
``valid_band_metrics``. The cafaeval parser was fixed in PR #75: the
previous parser keyed cafaeval's ``dfs_best`` output by namespace and
looked for a non-existent ``"metric"`` field, causing every
``f_micro_w`` value to be ``None``. The corrected parser keys by
metric-KIND (``"f_micro_w"``), reads the column of the same name, and
matches on the ``ns`` field.

GOA self-prior
~~~~~~~~~~~~~~

The GOA self-prior is the set of non-experimental (IEA and other
computational) GOA annotations for a query protein at t0. These are
leakage-safe: they are derived from sequence-based and orthology-based
evidence available at the start of the test period, not from future
experimental discoveries. The self-prior supplements the KNN candidate
list with terms that the embedding-based retrieval might miss. The
implementation lives in PROTEA core
(``apps/method_runtime/self_prior.py``), validated in slice
F-LAFA-SELFPRIOR.1. The universal booster trains on feature rows
exported by PROTEA, which may include self-prior candidates depending
on the export configuration; the lab does not re-implement the source.

Architecture
------------

The pipeline is split into discrete stages:

1. **Builder** (:mod:`protea_reranker_lab.builder`) reshapes a raw PROTEA
   dump into an experiment-ready parquet pair (``train.parquet`` and
   ``eval.parquet``) according to a :class:`~protea_reranker_lab.schemas.DatasetSpec`.

2. **Staging** (:mod:`protea_reranker_lab.staging`) performs multi-pass
   categorical encoding, protein-level routing (train/val split), and
   bucket-sort so that protein groups stay contiguous for LightGBM's
   ranking objective. The pooled variant
   (:mod:`protea_reranker_lab.pooled_staging`) extends this for
   multi-manifest inputs.

3. **Sequences** (:mod:`protea_reranker_lab.sequences`) expose sorted
   parquet buckets as :class:`lgb.Sequence` objects, enabling
   ``free_raw_data=True`` and capped peak RAM regardless of dataset size.

4. **Reranker** (:mod:`protea_reranker_lab.reranker`) wraps LightGBM
   training and streaming inference. The feature schema is imported from
   ``protea-contracts``, which is the canonical source of truth.

5. **Evaluate** (:mod:`protea_reranker_lab.evaluate`) computes
   protein-averaged Fmax on flat numpy arrays, mirroring PROTEA's own
   evaluation logic but without any pandas dependency.

6. **Compare** (:mod:`protea_reranker_lab.compare`) runs a per-protein
   paired bootstrap of Fmax between the trained booster and the
   KNN-only ``vote_count`` baseline to produce 95% confidence intervals
   and a one-sided p-value.

7. **Runner** (:mod:`protea_reranker_lab.runner`) wires all of the above
   into a single :func:`~protea_reranker_lab.runner.run_experiment` call
   that is invoked by the CLI (:mod:`protea_reranker_lab.train`).

8. **Universal runner** (:mod:`protea_reranker_lab.universal_runner`)
   orchestrates the full universal booster pipeline. Entry point:
   :func:`~protea_reranker_lab.universal_runner.run_universal`. Training
   helpers (staging, fitting, holdout evaluation) live in
   :mod:`protea_reranker_lab.universal_train`.

Feature schema
--------------

The feature set is pinned via ``protea-contracts``. The lab never adds
columns to ``ALL_FEATURES`` directly; changes must go through the
contracts package so that PROTEA's producer side stays in sync. The
universal booster adds two universal-specific axes (``plm_id`` as a
categorical feature, ``k_context`` as a float) that are injected per
source at staging time and are not present in the contracts feature list.

Reproducibility
---------------

Each run writes a ``run.json`` artefact that captures the resolved
hyperparameters, dataset lineage (spec and schema hashes from
:func:`~protea_reranker_lab.schemas.compute_schema_sha`), feature list,
timings, and final metrics. A ``spec.yaml`` is written at the start of
each run so that a crash still leaves a recoverable trace. The universal
booster run additionally writes ``pooled_staging_meta.json`` (row counts,
row budget, peak RSS) and ``plm_ablation_summary.json`` (plm_id feature
contribution).
