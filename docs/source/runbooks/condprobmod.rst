CondProbMod and soft Pmin/Pmax propagation
==========================================

This runbook covers the two hierarchy-aware levers added for slice R2.1 of
the roadmap-from-zero plan: ProtBoost's conditional-probability modelling
("CondProbMod"), the single biggest external ablation gain reported
(about +0.04 IA-Fmax), and the soft two-way Pmin/Pmax score-propagation pass.
Both operate on the GO DAG via a ``parent_map.json``.

What the two levers do
~~~~~~~~~~~~~~~~~~~~~~~~

CondProbMod (:mod:`protea_reranker_lab.condprobmod`)
    Training side: a child term is only learnable in the context of its
    parent, so the booster trains on the conditional target
    ``P(term | parent)`` and only on rows whose parent carries signal
    (a positive annotation or a non-zero prior prediction). Use
    :func:`~protea_reranker_lab.condprobmod.build_conditional_mask` to build
    that training mask. Inference side: the booster emits the per-edge
    conditional, and the marginal per-term probability is rebuilt by the
    parent-product recursion ``P(term) = P(term | parent) * P(parent)`` with
    :func:`~protea_reranker_lab.condprobmod.reconstruct_marginal_probs`.

Soft Pmin/Pmax (:mod:`protea_reranker_lab.soft_propagation`)
    A post-processing pass that blends the bottom-up Pmax direction (parent
    raised to max child) with the top-down Pmin direction (child capped at
    parent), default 0.7/0.3, then projects the blend back onto
    ``parent >= child`` consistency. Hard label propagation was already shown
    negligible for PROTEA; this is the soft, score-space variant. Entry point
    is :func:`~protea_reranker_lab.soft_propagation.soft_propagate_scores`.

Prerequisite: the parent map
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both levers need the GO is_a/part_of parent map for the dataset's ontology
snapshot. Export it once (with the PROTEA venv, which has ``psycopg``)::

    scripts/export_parent_map.py --manifest datasets/<frame>/manifest.json

That writes ``datasets/<frame>/parent_map.json``.

The ablation run (separate dispatch step)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

R2.1 ships the code and unit tests. The ablation that confirms the ~+0.04
target on the F0 validation frame is a separate run. The entry point is
``scripts/apply_hierarchy_postproc.py``, which takes a CAFA-format prediction
file plus the parent map and applies one or both passes.

Soft-propagation ablation (post-hoc, no retraining)::

    poetry run python scripts/apply_hierarchy_postproc.py \
        --pred preds.baseline.tsv \
        --parent-map datasets/<frame>/parent_map.json \
        --soft-prop --pmax-weight 0.7 \
        --out preds.softprop.tsv

CondProbMod marginal reconstruction (when the booster was trained on the
conditional target via ``build_conditional_mask``)::

    poetry run python scripts/apply_hierarchy_postproc.py \
        --pred preds.cond.tsv \
        --parent-map datasets/<frame>/parent_map.json \
        --condprobmod \
        --out preds.cond.marginal.tsv

Score baseline and treated files with cafaeval and read the IA-Fmax delta.
The full CondProbMod lever (conditional training plus marginal
reconstruction) is the one ProtBoost credits with about +0.04; the soft
propagation pass is a smaller, complementary consistency gain. Log the run
to MLflow (see :doc:`mlflow`) so the delta is tracked, not a silent script.
