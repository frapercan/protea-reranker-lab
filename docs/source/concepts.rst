Concepts and the lab workflow
=============================

This page tells the end-to-end story of the lab and then defines the three
ideas that everything else depends on: the IA-weighted ``f_micro_w``
selection metric, the LambdaRank group key, and the leakage discipline.
The formal metric definitions live in :doc:`metrics`; the stage-by-stage
mechanics with entrypoints live in :doc:`guide`.

.. contents:: On this page
   :depth: 2
   :local:

The candidate-ranking subproblem
--------------------------------

The reranker does not retrieve terms. PROTEA's KNN stage retrieves, for
each query protein, a list of candidate GO terms drawn from the nearest
reference proteins in PLM-embedding space. The reranker's only job is to
*reorder that list* so true terms float to the top. Every score, every
metric, and every comparison in the lab is therefore conditional on the
candidate set: the model can only be credited for ranking the candidates
KNN produced, never for recovering a term KNN never proposed. Keeping that
conditioning explicit is what makes the lab's results honest (see
:ref:`metrics-recovery-ceiling`).

The end-to-end workflow
-----------------------

A full pass runs from a frozen PROTEA export to a booster published back to
the platform registry:

.. code-block:: text

   export_research_dataset (PROTEA)
       train.parquet / eval.parquet / manifest.json
                 |
                 v
   1. staging      bucket-sorted parquet + labels.npy (group-contiguous)
   2. features     frozen feature schema pinned in protea-contracts
   3. training     LambdaMART or binary booster -> model.txt
   4. calibration  per-aspect isotonic / Platt map (optional)
   5. evaluation   cafaeval f_micro_w + numpy Fmax + bootstrap CI
                 |
                 v
   POST /reranker-models/import-by-reference (PROTEA registry)

**Pull a dataset.** PROTEA's ``export_research_dataset`` operation writes
``train.parquet``, ``eval.parquet``, and ``manifest.json`` for one cell
(a protein category crossed with a GO aspect, for example ``nk-mfo``). The
lab consumes those frozen files; it never queries the live stack.

**Stage.** :mod:`protea_reranker_lab.staging` streams the raw export into
LightGBM-ready parquet buckets, routing every row for a protein into the
same bucket so the protein's candidates stay contiguous. Contiguity is
what lets LightGBM form ranking groups without a global sort. The pooled
variant :mod:`protea_reranker_lab.pooled_staging` does the same across many
manifests at once for the universal booster, int-coding the heavy string
columns so memory stays bounded.

**Assemble features.** The feature columns are pinned by ``protea-contracts``
(:data:`~protea_contracts.ALL_FEATURES`). The lab never edits that list; a
schema change is a coordinated contracts bump so PROTEA's producer side
stays in lockstep. A schema digest is written into every run artifact, so a
silent column drift invalidates the cache instead of corrupting a
comparison.

**Train.** :mod:`protea_reranker_lab.reranker` wraps LightGBM. Buckets are
exposed as :class:`lgb.Sequence` objects through
:mod:`protea_reranker_lab.sequences` with ``free_raw_data=True``, so peak
RSS is capped regardless of dataset size. The default objective is
``lambdarank`` over protein groups; the per-cell champion uses a binary
objective. Optional IA sample weighting
(:mod:`protea_reranker_lab.ia_weighting`) prices high-information terms more
heavily during training.

**Calibrate.** :mod:`protea_reranker_lab.calibration` optionally fits one
isotonic or Platt map per GO aspect on the held-out window, turning raw
booster margins into comparable probabilities. The default path is
identity, so calibration never silently alters a run.
:mod:`protea_reranker_lab.hierarchical_correction` can then enforce the GO
DAG constraint that a parent term scores at least as high as its
highest-scoring child.

**Evaluate.** Three evaluators answer three questions: a fast numpy Fmax
proxy during training, the canonical IA-weighted ``f_micro_w`` through
cafaeval for selection, and a per-protein paired bootstrap against the KNN
baseline for confidence intervals. :mod:`protea_reranker_lab.recall`
separately reports the KNN retrieval ceiling so the reranker's lift is
never conflated with the candidate generator's recall limit.

**Publish.** A winning booster is promoted to PROTEA's ``RerankerModel``
registry through ``POST /reranker-models/import-by-reference``: PROTEA reads
the model by path and records its schema and manifest digests. The lab
never copies weights into the platform by hand.

Key concept: the IA-weighted ``f_micro_w`` metric
-------------------------------------------------

Selection is driven by an Information-Accretion (IA) weighted score, not a
plain Fmax. Information Accretion measures how much a term prediction adds
beyond what its parents already imply: a deep, rare term carries high IA
(high surprise), a generic near-root term carries IA close to zero.
Weighting precision and recall by IA rewards skill in the informative
region of the ontology, where plain Fmax is inflated by ancestor
propagation. The canonical evaluator is
:func:`protea_reranker_lab.evaluate.eval_f_micro_w`, which drives the
``cafaeval-protea`` subprocess with the band IA table and returns
``f_micro_w``. Because both the booster and the KNN baseline are scored on
the same candidate set, the *delta* between them is the publishable claim,
even though the absolute number is candidate-set restricted. The formal
definitions of IA, wFmax, and S_min, and why the internal proxy Fmax and
the cafaeval Fmax differ by roughly a factor of ten, are in :doc:`metrics`.

Key concept: the LambdaRank group key
--------------------------------------

LightGBM's ranking objective optimises an order *within a group*. The group
key defines which candidates compete against one another. For a single-cell
run the group is the protein: the model learns to rank a protein's
candidate terms relative to each other, never across proteins. For the
universal booster, which pools all 24 v226-lineage manifests
(8 PLM by K in {3, 5, 10}) into one model, the group key is widened to the
**four-tuple** ``(snapshot_pair, protein_accession, aspect, plm_id)``. That
scoping keeps each ranking query inside a single snapshot delta, aspect,
and PLM, so the model never ranks a candidate from one source against a
candidate from another and incoherent cross-source labels cannot pollute a
group. Staging guarantees the group key by bucketing on the protein and
keeping rows contiguous; the pooled stager carries the snapshot pair in the
bucket reserved columns. See :ref:`universal-staging`.

Key concept: leakage discipline
-------------------------------

The result on the line is a thesis number, so a booster must be selected
without ever seeing its test data. The train, validation, and test splits
follow a strict temporal order:

- **train**: snapshot pairs strictly before the validation holdout.
- **validation**: exactly the holdout snapshot pair (default ``"v220-v226"``),
  used to select hyperparameters and the champion.
- **test**: the reserved ``eval.parquet`` window (v226-v230), written at
  export time and never read during training or selection.

The protocol is *select on validation, score the test window once* on the
frozen champion. A degenerate band such as ``"v226-v227"`` is rejected for
validation: it contains only proteins newly annotated between two adjacent
GOA releases (single-digit ground-truth rows per cell), too sparse for
cafaeval to produce a meaningful ``f_micro_w``. Two further safeguards back
the temporal split: the negative-sampling audit
(:mod:`protea_reranker_lab.negative_sampling_audit`) records that negatives
are never encoded by lineage and rows are never replicated, and the
band-registry bridge (:mod:`protea_reranker_lab.band_registry_bridge`)
refuses to pair a cell with an IA table or ontology snapshot from a foreign
band.

Status and results
-------------------

The per-cell champion (binary objective, multi-seed, 2026-05-18) reaches a
selective-average cafaeval Fmax of **0.7291 +/- 0.0028** on
``bench-v1-K5-v226-lineage`` (NK+LK cells, three seeds). All six NK+LK
paired-bootstrap confidence intervals are strictly positive at 95 percent
(10000 iterations per cell). The selective-deployment policy promotes the
binary champion on NK and LK cells and keeps PK cells on the KNN baseline
(PK gains are policy-zero, per ADR D34). This is the number cited in
Chapter 6 of the doctoral thesis.

The universal booster (F-RERANK-UNIVERSAL) is a proof of concept that
replaces the per-cell models with one aspect-conditioned, IA-weighted
LambdaMART model. It beats the ``prot_t5 K3`` KNN baseline on NK+LK mean
``f_micro_w`` on the held-out ``"v220-v226"`` band; its absolute number is
candidate-set restricted and awaits a clean v227-lineage recompute before
publication (see :doc:`universal_reranker`).

Two earlier figures must not be cited in place of 0.7291. An Fmax of 0.4562
came from a training parquet that replicated the GO-category column across
snapshots, inflating the apparent training signal without introducing
temporal leakage; the mechanism is documented in PROTEA memory
``project_anc2vec_leakage_mechanism``. The intermediate LB.2 estimate
(0.6215 +/- 0.0014, leakage-fixed) is superseded by the binary-objective
champion. The full per-cell number table is in :doc:`metrics`.

Reproducibility artifacts
-------------------------

Every run writes a ``spec.yaml`` at the start (so a crash still leaves a
recoverable trace) and a ``run.json`` at the end capturing resolved
hyperparameters, dataset lineage (spec and schema digests from
:func:`~protea_reranker_lab.schemas.compute_schema_sha`), the feature list,
timings, and final metrics. The universal booster additionally writes
``pooled_staging_meta.json`` (row counts, row budget, peak RSS) and
``plm_ablation_summary.json`` (the contribution of ``plm_id`` to
``f_micro_w``).
