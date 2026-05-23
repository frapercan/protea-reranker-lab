Overview
========

**protea-reranker-lab** is a self-contained experimentation environment
for the LightGBM-based GO-term reranker used in PROTEA. It takes the
raw 56-feature parquet dumps produced by PROTEA's
``export_research_dataset`` operation and trains a streaming ranking
model without ever materialising full data frames in memory.

Published champion
------------------

The current publishable result (multi-seed, leakage-fixed, 2026-05-17)
is a selective-average cafaeval Fmax of **0.6215 +/- 0.0014** on
``bench-v1-K5-v226-lineage`` (NK+LK cells, 3 seeds). All six NK+LK
paired-bootstrap confidence intervals are strictly positive at 95%
(N=10000 bootstrap iterations per cell). This figure is cited in
Chapter 6 of the doctoral thesis.

An earlier number (0.4562) arose from a train-parquet construction
issue where the GO category column was replicated across snapshots for
the same protein-term pair, inflating the effective training signal
without introducing temporal leakage. The fix is tracked in PROTEA
memory key ``project_anc2vec_leakage_mechanism``. Do not cite 0.4562
in any published output.

Architecture
------------

The pipeline is split into discrete stages:

1. **Builder** (:mod:`protea_reranker_lab.builder`) reshapes a raw PROTEA
   dump into an experiment-ready parquet pair (``train.parquet`` and
   ``eval.parquet``) according to a :class:`~protea_reranker_lab.schemas.DatasetSpec`.

2. **Staging** (:mod:`protea_reranker_lab.staging`) performs multi-pass
   categorical encoding, protein-level routing (train/val split), and
   bucket-sort so that protein groups stay contiguous for LightGBM's
   ranking objective.

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

Feature schema
--------------

The feature set is pinned via ``protea-contracts``. The lab never adds
columns to ``ALL_FEATURES`` directly; changes must go through the
contracts package so that PROTEA's producer side stays in sync.

Reproducibility
---------------

Each run writes a ``run.json`` artefact that captures the resolved
hyperparameters, dataset lineage (spec and schema hashes from
:func:`~protea_reranker_lab.schemas.compute_schema_sha`), feature list,
timings, and final metrics. A ``spec.yaml`` is written at the start of
each run so that a crash still leaves a recoverable trace.
