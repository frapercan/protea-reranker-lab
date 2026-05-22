Overview
========

**protea-reranker-lab** is a self-contained experimentation environment
for the LightGBM-based GO-term reranker used in PROTEA. It takes the
raw 56-feature parquet dumps produced by PROTEA's
``export_research_dataset`` operation and trains a streaming ranking
model without ever materialising full data frames in memory.

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

6. **Runner** (:mod:`protea_reranker_lab.runner`) wires all of the above
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
