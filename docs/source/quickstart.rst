Quickstart
==========

This page gets you from a clone to a trained, evaluated booster. For the
workflow as a story see :doc:`concepts`; for the stage-by-stage mechanics
see :doc:`guide`; for the full publish-back flow see the project ``README``.

.. contents:: On this page
   :depth: 2
   :local:

Install
-------

Python 3.12 or later is required.

.. code-block:: bash

   git clone https://github.com/frapercan/protea-reranker-lab.git
   cd protea-reranker-lab
   git checkout develop

   pip install -e .          # runtime only
   pip install -e ".[dev]"   # + ruff / mypy / sphinx / pytest

Or with Poetry, which resolves the pinned ``protea-contracts`` commit:

.. code-block:: bash

   poetry install
   poetry run pytest -q tests/

Pull a dataset from PROTEA
--------------------------

The lab consumes frozen exports; it never queries the live stack. Ask
PROTEA to produce one through the REST surface (never via ad-hoc curl to
internal endpoints):

.. code-block:: bash

   curl -s -X POST http://localhost:3000/api/v1/datasets \
     -H "Content-Type: application/json" \
     -d '{"operation": "export_research_dataset",
          "payload": {"cell": "nk-mfo",
                      "train_versions": [160, 200, 210, 215, 220],
                      "test_versions": [230],
                      "k": 5,
                      "embedding_config_id": "<uuid>"}}'

Download the resulting ``train.parquet``, ``eval.parquet``, and
``manifest.json`` into ``datasets/<name>/``, then validate the manifest:

.. code-block:: bash

   python scripts/validate_manifest.py --manifest datasets/<name>/manifest.json

Train a single cell
--------------------

.. code-block:: bash

   python -m protea_reranker_lab.train \
     --dataset datasets/bench-v1-K5-v226-lineage \
     --cell nk-mfo \
     --objective lambdarank \
     --num-boost-round 5000 \
     --early-stopping-rounds 50 \
     --seed 42 \
     --wandb-mode disabled \
     --output-dir runs/nk-mfo-seed42

Training writes ``model.txt``, ``run.json`` (with ``test_fmax``), and
``predictions.parquet`` into the output directory.

Train the universal booster
----------------------------

A single aspect-conditioned, IA-weighted LambdaMART model over all 24
v226-lineage manifests:

.. code-block:: bash

   python scripts/run_universal_booster.py \
     --datasets-root datasets \
     --val-holdout-snapshot v220-v226 \
     --out runs/universal

See :doc:`universal_reranker` for the design and the held-out band
protocol.

Evaluate without retraining
---------------------------

Recompute the fast numpy proxy metric on an existing predictions file:

.. code-block:: python

   import pyarrow.parquet as pq
   from protea_reranker_lab.evaluate import fmax_per_protein_group

   table = pq.read_table(
       "runs/nk-mfo-seed42/predictions.parquet",
       columns=["label", "score", "group_size"],
   )
   print(
       fmax_per_protein_group(
           table["score"].to_numpy(),
           table["label"].to_numpy(),
           table["group_size"].to_numpy(),
       )
   )

For the canonical IA-weighted selection metric, use
:func:`~protea_reranker_lab.evaluate.eval_f_micro_w` with a band IA table
resolved through
:func:`~protea_reranker_lab.band_registry_bridge.resolve_band_artifacts`.

Compare against the KNN baseline
--------------------------------

.. code-block:: bash

   python scripts/run_bootstrap_phase.py \
     --predictions runs/nk-mfo-seed42/predictions.parquet \
     --source-eval datasets/bench-v1-K5-v226-lineage/eval.parquet \
     --cell nk-mfo \
     --n-iter 10000 \
     --workdir /tmp/bootstrap \
     --out runs/nk-mfo-seed42/bootstrap.json

The paired bootstrap reports a 95 percent confidence interval on the
per-protein Fmax delta and a one-sided p-value. A strictly positive
interval is the publishable evidence that the reranker beats the
candidate generator on that cell.

Local checks
------------

Run the same gates CI runs before opening a pull request:

.. code-block:: bash

   ruff check src scripts
   mypy
   poetry run pytest -q tests/
   python scripts/check_smells.py --target src
   poetry run sphinx-build -W -b html docs/source docs/_build/html
