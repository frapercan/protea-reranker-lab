protea-reranker-lab
===================

**protea-reranker-lab** is the offline laboratory where PROTEA's GO-term
reranker is researched, trained, and selected before it is promoted into
the platform. PROTEA's KNN retrieval already produces competitive
candidate rankings; the reranker is a LightGBM model that reorders those
candidates to lift the ranking quality. The lab is where that model is
built and proven.

Its defining property is decoupling. Training over millions of
protein-term candidate rows inside the live PROTEA stack would tie up the
API server, materialise large data frames, and compete for GPU time with
embedding jobs. Instead PROTEA exports a *frozen* parquet snapshot once,
and the lab iterates on it entirely offline, streaming through sorted
parquet buckets so peak memory stays bounded no matter how large the
dataset is. Nothing in the lab talks to a live database or queue.

The role and the problem
------------------------

The lab solves one problem end to end: **train and select a booster that
is provably better than the KNN baseline, leakage-clean, before promoting
it back to PROTEA.** That single sentence carries the three constraints
that shape every module here:

- *Train.* Fit a LightGBM ranker (or binary classifier) over the frozen
  candidate-ranking dataset without ever loading it whole.
- *Select.* Score candidates with the same IA-weighted metric the CAFA
  protocol uses, on a held-out temporal window, so the chosen booster
  earned its place rather than overfitting the selection set.
- *Promote.* Publish the winning model back to PROTEA by reference (a
  registry import), never by copying weights into the platform by hand.

The reranker operates on a *candidate-ranking subproblem*: it reorders the
terms KNN already retrieved, so its scores and metrics are conditional on
that candidate set. Keeping that conditioning explicit, and never letting
future labels leak into training or selection, is the discipline the whole
codebase is organised around. See :doc:`concepts` for the full picture.

What lives here
---------------

The pipeline is a sequence of small, testable modules. Reading order
roughly follows the data:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Stage
     - Module
   * - Reshape a raw export
     - :mod:`protea_reranker_lab.builder`
   * - Single-manifest staging
     - :mod:`protea_reranker_lab.staging`
   * - Pooled multi-manifest staging
     - :mod:`protea_reranker_lab.pooled_staging`
   * - Stream buckets to LightGBM
     - :mod:`protea_reranker_lab.sequences`
   * - Train / predict
     - :mod:`protea_reranker_lab.reranker`
   * - IA sample weighting
     - :mod:`protea_reranker_lab.ia_weighting`
   * - Per-aspect calibration
     - :mod:`protea_reranker_lab.calibration`
   * - Hierarchical correction
     - :mod:`protea_reranker_lab.hierarchical_correction`
   * - Numpy and IA-weighted eval
     - :mod:`protea_reranker_lab.evaluate`
   * - KNN retrieval ceiling
     - :mod:`protea_reranker_lab.recall`
   * - Paired bootstrap vs baseline
     - :mod:`protea_reranker_lab.compare`, :mod:`~protea_reranker_lab.bootstrap`
   * - Band / IA artifact bridge
     - :mod:`protea_reranker_lab.band_registry_bridge`
   * - Per-cell orchestration
     - :mod:`protea_reranker_lab.runner`
   * - Universal booster orchestration
     - :mod:`protea_reranker_lab.universal_runner`

If you are new here, read :doc:`concepts` for the workflow as a story,
then :doc:`quickstart` to run it, then :doc:`guide` for stage detail. The
:doc:`metrics`, :doc:`ia_aligned_training`, and :doc:`universal_reranker`
pages are deep dives. The :doc:`reference/index` documents every module.

.. toctree::
   :maxdepth: 2
   :caption: The lab

   concepts
   quickstart
   guide

.. toctree::
   :maxdepth: 2
   :caption: Deep dives

   metrics
   ia_aligned_training
   universal_reranker

.. toctree::
   :maxdepth: 1
   :caption: Reference

   reference/index
   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
