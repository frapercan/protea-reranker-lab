IA-aligned training (palanca 1)
===============================

This page documents the Information-Accretion (IA) sample-weighting lever
("palanca 1") that pushes IA *into* reranker training, the configuration
surface, and the honest probe result on the ``lk-bpo`` cell. The formal IA
definition, provenance, and the four-metric taxonomy live in
:doc:`metrics`. The decision record is
`ADR D40 <https://github.com/frapercan/protea-reranker-lab/blob/develop/docs/adr/D40-ia-aligned-training.md>`_.

.. contents:: On this page
   :depth: 2
   :local:

Motivation
----------

The v26/v27-binary champion trains with ``objective: binary`` and uniform
sample weights. Its headline cafaeval Fmax (``prop=fill``, ``norm=cafa``) is
dominated by ancestor propagation, which regales the shallow, frequent terms
the model barely has to work for. The honest metric is the IA-weighted Fmax
(wFmax) and the family of S_min distances, which downweight shallow terms and
reward skill in the deep, informative region (see
:ref:`metrics-ia-provenance`). Palanca 1 asks whether re-pricing training
rows by ``IA(go)`` lifts that honest metric.

Configuration
-------------

The lever is exposed on :class:`protea_reranker_lab.reranker.TrainConfig` and
on the ``scripts/run.py`` / ``train.py`` CLI:

``ia_weighting`` (default ``none``)
    Weighting mode. ``none`` keeps the historical uniform-weight path
    byte-for-byte (the weight array is ``None``). ``positives`` weights only
    positive rows by ``1 + ia_scale * IA(go)`` and leaves negatives at
    ``1.0``. ``all`` weights every row (positives and negatives) by
    ``1 + ia_scale * IA(go)``.

``ia_path`` (default ``datasets/ia/IA-swissprot-exp-v227.txt``)
    The IA table (two columns, ``GO:xxxxxxx\t<IA>``). Terms absent from the
    table contribute ``IA = 0`` (neutral weight ``1.0``); malformed or
    negative IA values are clamped to ``0.0``.

``ia_scale`` (default ``1.0``)
    Linear scale on IA: ``weight = 1 + ia_scale * IA(go)``.

Mechanics: row-aligned weights
------------------------------

Sample weights must align row-for-row with ``labels.npy``, which the
streaming staging pipeline produces by bucketing rows per protein and
stable-sorting within each bucket. When ``ia_weighting`` is not ``none``,
:func:`protea_reranker_lab.staging.stage_for_training` threads ``go_term_id``
through the route-and-sort passes and emits a row-aligned ``go_terms.npy`` per
split. :func:`protea_reranker_lab.runner.run_experiment` loads it, maps each
row to ``IA(go)`` via
:func:`protea_reranker_lab.ia_weighting.ia_weights`, and passes the weight
vector into both ``lgb.Dataset`` calls. The eval split never carries go terms
(IA weighting applies to training only). The default ``none`` path neither
carries go terms nor changes the staging schema.

Probe result (lk-bpo, seed 42, bench-v1-K5-v226-lineage)
--------------------------------------------------------

Three reranker arms (v26-binary recipe, all else equal) plus the KNN
baseline, each scored with cafaeval under the LAFA protocol (``prop=fill``,
``norm=cafa``, ``no_orphans``, ``ia=IA-swissprot-exp-v227.txt``). Driver:
``experiments/lafa_ia_palanca1/probe_lk_bpo.py``.

.. list-table::
   :header-rows: 1

   * - arm
     - cafaeval Fmax
     - wFmax
     - S_min
     - d wFmax vs KNN
   * - unweighted
     - 0.6485
     - **0.5945**
     - 7.057
     - +0.1924
   * - ia_positives
     - 0.6417
     - 0.5872
     - 7.103
     - +0.1851
   * - ia_all
     - 0.6466
     - 0.5890
     - 7.089
     - +0.1868
   * - knn baseline
     - 0.4639
     - 0.4021
     - 11.309
     - n/a

Verdict and honesty
-------------------

The reranker beats the KNN baseline by a wide margin on the weighted metric
(+0.19 wFmax, well above the F-LAFA-IA.0 reference of +0.0785), so reranking
is clearly worthwhile. But IA sample weighting did **not** lift wFmax over the
plain unweighted reranker: both IA arms dipped slightly (0.5872 / 0.5890
versus 0.5945) and worsened S_min (7.10 / 7.09 versus 7.06; lower is better).
The gate verdict is **STOP**: palanca 1 is not fanned out to the 24-cell grid.

This is the result the brief flagged as a genuine possibility (honesty over
headline): a flat per-row ``weight=`` is a blunt instrument for a per-protein
ranking metric, and binary log-loss with ``neg_pos_ratio=10`` already
concentrates the gradient on positives. The recommended next slice is
palanca 2 (lambdarank with IA-encoded ``label_gain``), which tells the ranker
*directly* to lift high-IA terms above generic and negative ones. The
palanca-1 machinery (the ``weight=`` hook, the IA loader, the row-aligned
``go_terms.npy``) stays in place and feeds palanca 2's relevance encoding.
