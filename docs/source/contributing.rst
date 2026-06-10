Contributing
============

The lab is a research codebase with a publishable result on the line, so
the bar for a change is correctness and reproducibility rather than
feature breadth.

Branching and review
--------------------

- All work targets ``develop``; ``main`` tracks tagged releases.
- Branch, commit, open a pull request against ``develop``. Never push
  straight to ``develop`` or ``main``.
- Keep the feature schema owned by ``protea-contracts``: renaming or
  adding a feature is a coordinated pull-request pair plus a contracts
  version bump, never a local edit to the feature list.

Local gates
-----------

Run every gate that CI runs before opening a pull request:

.. code-block:: bash

   ruff check src scripts
   mypy
   poetry run pytest -q tests/
   python scripts/check_smells.py --target src
   poetry run sphinx-build -W -b html docs/source docs/_build/html

The ``mypy`` configuration lives in ``pyproject.toml``; the native deps
(``pyarrow``, ``lightgbm``, ``wandb``) are treated as untyped so their
import-time namespaces do not raise spurious ``attr-defined`` errors.

Smell budget
------------

``scripts/check_smells.py`` enforces a frozen budget recorded in
``.smell-baseline.json`` (file length, class length, method length,
parameter count). New code must not introduce or worsen an offender.
After a genuine refactor that legitimately changes the counts, refresh
the baseline:

.. code-block:: bash

   python scripts/check_smells.py --write-baseline

Prose linters
-------------

- The reranker-token linter rejects bare reranker shorthand in prose;
  use the axis-tuple form or a GOA snapshot name (``v226``, ``v230``).
- The dataset-name linter rejects untagged
  ``bench-v1-K{k}-v226-lineage`` references that omit the PLM suffix.
- Published prose (README, docs, ADRs, CHANGELOG) must not contain
  em-dashes; use a period, comma, or parentheses instead.

Tests that guard the thesis numbers
------------------------------------

Three suites under ``tests/`` protect the bits whose silent breakage
would invalidate the Chapter 6 results:

- ``test_schema_sha_determinism.py`` keeps the 12-hex ``schema_sha``
  order-independent and stable across dict and parquet roundtrips.
- ``test_golden_parquet_roundtrip.py`` asserts reserved columns, the
  ``protea-contracts`` feature membership, numeric dtype, and binary
  labels on a golden fixture.
- ``test_train_eval_split_determinism.py`` pins the train/val/eval split
  for both the ``protein_group`` and ``temporal`` strategies, including
  the negative-downsampling branch.

Documentation
-------------

Docstrings use the Google or NumPy style (both are enabled via
``sphinx.ext.napoleon``). Every public module is autodoc-ed under
:doc:`reference/index`; a new module only needs a one-line
``.. automodule::`` stub and an entry in the reference toctree. Build
with ``-W`` so a broken cross-reference fails locally rather than in CI.

Artefacts
---------

Dataset dumps (``datasets/``) and run outputs (``runs/``) are
git-ignored. Never commit large parquet files or trained model binaries;
publish boosters back to PROTEA through
``POST /reranker-models/import-by-reference`` instead.
