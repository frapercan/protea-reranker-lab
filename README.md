# protea-reranker-lab

[![CI](https://github.com/frapercan/protea-reranker-lab/actions/workflows/quality.yml/badge.svg)](https://github.com/frapercan/protea-reranker-lab/actions/workflows/quality.yml)
[![Documentation](https://img.shields.io/readthedocs/protea-reranker-lab.svg)](https://protea-reranker-lab.readthedocs.io)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/protea-reranker-lab.svg)](https://pypi.org/project/protea-reranker-lab/)

Offline LightGBM training laboratory for the PROTEA GO-term reranker.
The lab consumes frozen feature datasets exported from PROTEA, fits a
streaming ranking model without ever loading a full DataFrame into
memory, evaluates per ontology-aspect cell, and publishes winning
boosters back to PROTEA's `RerankerModel` registry.

**Status:** production-grade results pipeline. The current champion
(binary-objective, multi-seed, 2026-05-18) achieves NK+LK cafaeval Fmax
**0.7291 +/- 0.0028** on `bench-v1-K5-v226-lineage` (3 seeds, paired
bootstrap 95% CI strictly positive vs KNN baseline on all six NK+LK
cells). Results are published to the doctoral thesis (Chapter 6).

<!-- protea-stack:start -->

## Repositories in the PROTEA stack

Single source of truth: [`docs/source/_data/stack.yaml`](https://github.com/frapercan/PROTEA/blob/develop/docs/source/_data/stack.yaml) in PROTEA. Run `python scripts/sync_stack.py` to regenerate this block.

| Repo | Role | Status | Summary |
|------|------|--------|---------|
| [PROTEA](https://github.com/frapercan/PROTEA) | Platform | `active` | Backend platform. Hosts the ORM, job queue, FastAPI surface, frontend, and orchestration. |
| [protea-contracts](https://github.com/frapercan/protea-contracts) | Contracts | `active` | Shared contract surface. ABCs, pydantic payloads, feature schema, schema_sha. Imported by every other repo. |
| [protea-method](https://github.com/frapercan/protea-method) | Inference | `active` | Pure inference path (KNN, feature compute, reranker apply). LAFA inference layer; publishes to DockerHub. |
| [protea-sources](https://github.com/frapercan/protea-sources) | Source plugin | `active` | Annotation source plugins (GOA, QuickGO, UniProt, InterPro). Discovered via Python entry_points. |
| [protea-runners](https://github.com/frapercan/protea-runners) | Runner plugin | `active` | Experiment runner plugins (LightGBM, KNN, baseline). Discovered via Python entry_points. |
| [protea-backends](https://github.com/frapercan/protea-backends) | Backend plugin | `active` | Protein language model embedding backends (ESM family, T5/ProstT5, Ankh, ESM3-C). Discovered via Python entry_points. |
| **protea-reranker-lab** (this repo) | Lab | `active` | LightGBM reranker training lab. Pulls datasets from PROTEA, trains boosters, publishes them back via /reranker-models/import-by-reference. |
| [cafaeval-protea](https://github.com/frapercan/cafaeval-protea) | Evaluator | `active` | Standalone fork of cafaeval (CAFA-evaluator-PK) with the PK-coverage fix and a bit-exact parity guarantee against the upstream. |

<!-- protea-stack:end -->

## What and why

PROTEA's KNN candidate retrieval already produces competitive GO-term
rankings (vote-count baseline). A LightGBM reranker trained on the
56-feature schema defined in `protea-contracts` brings a statistically
significant improvement over that baseline.

Training on millions of protein-term pairs inside the full PROTEA stack
would require materialising large DataFrames, tying up the API server,
and sharing scarce GPU resources with embedding jobs. This lab decouples
all of that: PROTEA exports a *frozen* parquet snapshot once, and the
lab iterates on hyperparameters and ablations entirely offline, reading
through sorted parquet buckets as `lgb.Sequence` objects to keep peak
RSS bounded below 15 GB even on the largest cell.

**Current per-cell champion (binary-objective, multi-seed, 2026-05-18):**
NK+LK selective average cafaeval Fmax **0.7291 +/- 0.0028** on
`bench-v1-K5-v226-lineage` (3 seeds). All six NK+LK paired-bootstrap
confidence intervals vs the KNN baseline are strictly positive at 95%
(N=10000). Selective deployment policy: NK+LK cells use the binary-objective
champion; PK cells remain on the KNN baseline (PK gains are policy-zero; see
ADR D34). This is the publishable number for Chapter 6 of the doctoral thesis.

**Universal booster (F-RERANK-UNIVERSAL, PoC, 2026-06-08):** a single
aspect-conditioned, K-augmented, IA-weighted LambdaMART model trained over all
24 v226-lineage manifests (8 PLM x K{3,5,10}) replaces the per-cell phase3a
models. The PoC validates on the held-out `"v220-v226"` band and beats the
`prot_t5 K3` KNN baseline on NK+LK mean `f_micro_w`. The absolute number is
candidate-set restricted; a clean v227-lineage recompute is deferred before
publication. See [ADR D41](docs/adr/D41-universal-reranker.md) and
[universal reranker docs](docs/source/universal_reranker.rst).

The earlier LB.2 estimate (0.6215 +/- 0.0014) is superseded by the per-cell
champion and should not be cited in place of 0.7291 in new writing. See the
[Leakage history](#leakage-history) note for the full number genealogy and
why 0.4562 must not be cited.

## Place in the stack

```
PROTEA (export_research_dataset)
    └── datasets/bench-v1-K5-v226-lineage/
            ├── train.parquet
            ├── eval.parquet
            └── manifest.json
                    |
                    v
        protea-reranker-lab
            ├── staging  → sorted bucket parquets + labels.npy
            ├── train    → LightGBM booster (model.txt)
            ├── eval     → numpy Fmax per cell
            └── compare  → paired bootstrap CI vs KNN baseline
                    |
                    v
        POST /reranker-models/import-by-reference
            → PROTEA RerankerModel registry
```

## Documentation

Full documentation is published to ReadTheDocs at
https://protea-reranker-lab.readthedocs.io. It lives under `docs/`
and builds with Sphinx:

```bash
poetry install
poetry run sphinx-build -W -b html docs/source docs/_build/html
```

It covers a quickstart, a stage-by-stage workflow tour (pooled staging,
features, LambdaMART training, calibration, evaluation), the evaluation
metrics and IA-weighted `f_micro_w` evaluator, the universal booster, a
contributing guide, and a full per-module API reference. Release notes
are tracked in [`CHANGELOG.md`](CHANGELOG.md).

## Install

Python 3.12 or later is required.

```bash
git clone https://github.com/frapercan/protea-reranker-lab.git
cd protea-reranker-lab
git checkout develop

pip install -e .          # runtime only
pip install -e ".[dev]"   # + ruff / mypy / sphinx
```

## Dataset pull, train, evaluate, import flow

### Step 1: pull dataset from PROTEA

Dispatch an `export_research_dataset` job through the PROTEA REST API.
Never pull via ad-hoc curl; use the `POST /datasets` endpoint:

```bash
curl -s -X POST http://localhost:3000/api/v1/datasets \
  -H "Content-Type: application/json" \
  -d '{"operation": "export_research_dataset",
       "payload": {"cell": "nk-mfo",
                   "train_versions": [160,200,210,215,220],
                   "test_versions": [230],
                   "k": 5,
                   "embedding_config_id": "<uuid>"}}'
```

The job writes `train.parquet`, `eval.parquet`, and `manifest.json`
to PROTEA's storage. Download those three files into
`datasets/<your-dataset-name>/` before running the lab.

### Step 2: train

The CLI entry-point is `protea_reranker_lab.train`:

```bash
python -m protea_reranker_lab.train \
  --dataset datasets/bench-v1-K5-v226-lineage \
  --cell nk-mfo \
  --objective lambdarank \
  --num-boost-round 5000 \
  --early-stopping-rounds 50 \
  --seed 42 \
  --wandb-mode disabled \
  --output-dir runs/nk-mfo-seed42
```

Or run a full study phase (sequential, resumable):

```bash
python scripts/build_study_specs.py   # generate YAMLs under experiments/
python scripts/run_study.py f1        # 27 replication specs
python scripts/run_study.py f2        # 39 ablation specs (leave-one-family-out)
python scripts/run_study.py f4        # 27 hparam-grid specs
```

### Step 3: evaluate

Evaluation runs automatically at the end of each training call and
writes `test_fmax` to `run.json`. To recompute metrics on an existing
predictions file without re-training:

```python
from pathlib import Path
import pyarrow.parquet as pq, numpy as np
from protea_reranker_lab.evaluate import fmax_per_protein_group

table = pq.read_table("runs/nk-mfo-seed42/predictions.parquet",
                      columns=["label", "score", "group_size"])
labels = table["label"].to_numpy()
scores = table["score"].to_numpy()
groups = table["group_size"].to_numpy()
print(fmax_per_protein_group(scores, labels, groups))
```

Aggregate all phase results into a summary table:

```bash
python scripts/summarise_study.py
# writes runs/study_<name>/SUMMARY.md
```

### Step 4: compare vs KNN baseline

Run the paired bootstrap to obtain confidence intervals:

```bash
python scripts/run_bootstrap_phase.py \
  --predictions runs/nk-mfo-seed42/predictions.parquet \
  --source-eval datasets/bench-v1-K5-v226-lineage/eval.parquet \
  --cell nk-mfo \
  --n-iter 10000 \
  --workdir /tmp/bootstrap \
  --out runs/nk-mfo-seed42/bootstrap.json
```

### Step 5: import winner to PROTEA

```bash
curl -s -X POST http://localhost:3000/api/v1/reranker-models/import-by-reference \
  -H "Content-Type: application/json" \
  -d '{"model_path": "/abs/path/to/runs/nk-mfo-seed42/model.txt",
       "manifest_sha": "<from manifest.json>",
       "schema_sha": "<from run.json>",
       "cell": "nk-mfo",
       "metrics": {"test_fmax": 0.7528}}'
```

## Leakage history

An earlier cafaeval Fmax number (0.4562) appeared in internal records
from a run using the v226-series GOA snapshot. That figure came from a
multi-snapshot training parquet where the GO category column was
replicated across snapshots for the same protein-term pair. The
replication made the train-set size appear artificially large and
inflated the training signal without introducing temporal leakage (no
future labels bled into training). The mechanism is documented in
PROTEA memory key `project_anc2vec_leakage_mechanism`.

The fix (anc2vec retrofix, multi-seed validation) produced the
**0.6215 +/- 0.0014** LB.2 estimate. A subsequent binary-objective training
run (three seeds, binary classification objective instead of lambdarank)
produced the current champion **0.7291 +/- 0.0028**. The progression is:
0.4562 (artefact, do not cite) to 0.6215 (LB.2, superseded) to 0.7291
(binary-objective champion, current; cite this for all new writing on bench-v1-K5-v226-lineage NK+LK).

The selective-rerank decision (ADR D34) is documented in
`docs/decisions/D34-selective-rerank-policy.md` and the formal run
records are in `experiments/study-selective-rerank-K10-v226/`.

## Repo layout

```
protea-reranker-lab/
├── datasets/                  # frozen feature dumps (git-ignored)
│   └── <name>/
│       ├── train.parquet
│       ├── eval.parquet
│       ├── manifest.json
│       └── README.md          # auto-generated per-dataset README
├── dataset_cards/             # per-PLM HuggingFace-style dataset cards
│   ├── esm2_150m_card.md
│   ├── esm2_650m_card.md
│   └── ...                    # one card per PLM (8 total)
├── src/protea_reranker_lab/
│   ├── train.py           # per-cell CLI entry-point
│   ├── evaluate.py        # numpy Fmax
│   ├── compare.py         # bootstrap comparison namespace
│   ├── staging.py         # single-manifest bucket-sort + cell filter
│   ├── pooled_staging.py  # multi-manifest pooled staging (int-code OOM fix)
│   ├── bootstrap.py       # paired bootstrap CIs
│   ├── runner.py          # ExperimentSpec orchestrator
│   ├── universal_runner.py# universal booster orchestrator (F-RERANK-UNIVERSAL)
│   ├── universal_train.py # training helpers + holdout-band evaluator
│   ├── multi_source.py    # MultiManifestSpec + ManifestSource
│   ├── reranker.py        # LightGBM fit / predict + IA-feval callback
│   ├── sequences.py       # lgb.Sequence on parquet buckets
│   ├── builder.py         # streaming reshape
│   ├── data.py            # PyArrow streaming primitives
│   ├── experiment.py      # ExperimentSpec / DatasetRef
│   ├── schemas.py         # ManifestV1 + schema_sha
│   ├── calibration.py     # per-aspect isotonic calibration
│   ├── hierarchical_correction.py  # parent >= max-child score correction
│   └── ia_weighting.py    # IA sample weights + feval callback
├── scripts/
│   ├── validate_manifest.py        # F-DATA-PACK.1: manifest schema + hash validator
│   ├── generate_dataset_readme.py  # F-DATA-PACK.2: per-dataset README generator
│   ├── run_universal_booster.py    # universal booster CLI (F-RERANK-UNIVERSAL)
│   └── ...                         # other CLI drivers (run.py, run_study.py, ...)
├── docs/
│   ├── adr/
│   │   ├── D34-selective-rerank-resurrection.md
│   │   ├── D39-f-data-pack-fair-dataset-packaging.md
│   │   ├── D40-ia-aligned-training.md
│   │   └── D41-universal-reranker.md   # universal booster design decisions
│   ├── dataset_provenance.md  # F-DATA-PACK.4: FAIR/coverage provenance document
│   └── source/
│       ├── index.rst         # narrative intro + module map
│       ├── concepts.rst      # workflow story + key concepts
│       ├── quickstart.rst
│       ├── guide.rst         # stage-by-stage mechanics
│       ├── metrics.rst
│       ├── ia_aligned_training.rst
│       └── universal_reranker.rst      # universal booster documentation
└── experiments/               # YAML spec files
```

## Dataset packaging (F-DATA-PACK)

The F-DATA-PACK loop materialises the 24-dataset grid produced by FARM-EXP.13
as FAIR-compliant, citable research artefacts. Four slices have landed on
`develop` (lab PRs #48, #49, #50, #51); a fifth (Zenodo deposit, F-DATA-PACK.5)
is pending:

| Slice | PR | Surface |
|-------|----|---------|
| F-DATA-PACK.1 | #48 | `scripts/validate_manifest.py`: schema + content-hash validator wired into CI |
| F-DATA-PACK.2 | #49 | `scripts/generate_dataset_readme.py`: auto-generates `datasets/<name>/README.md`; 11 cells emitted |
| F-DATA-PACK.3 | #50 | `dataset_cards/<plm>_card.md`: 8 per-PLM HuggingFace-style dataset cards |
| F-DATA-PACK.4 | #51 | `docs/dataset_provenance.md`: FAIR checklist, split methodology, PCA fit policy, leakage note |
| F-DATA-PACK.5 | pending | Zenodo/HuggingFace Hub deposit of the 24-dataset grid |

**Validate a manifest before training:**

```bash
python scripts/validate_manifest.py --manifest datasets/<name>/manifest.json
```

**Re-generate a per-dataset README after pulling a new dataset:**

```bash
python scripts/generate_dataset_readme.py --dataset datasets/<name>
```

**Provenance and FAIR compliance:** see `docs/dataset_provenance.md` for the full
data lineage (GOA window, PCA fit policy, leakage-free note, FAIR checklist).

**Architecture decision records:**

- [ADR D34](docs/adr/D34-selective-rerank-resurrection.md): selective rerank policy
  (LightGBM champion design)
- [PROTEA ADR D35](https://github.com/frapercan/PROTEA/blob/develop/docs/source/adr/D35-canonical-8plm-embedding-configs.rst):
  canonical 8-PLM embedding config IDs (embedding_config_id table)
- [PROTEA ADR D38](https://github.com/frapercan/PROTEA/blob/develop/docs/source/adr/D38-neural-head-deferred-dataset-pack-pivot.rst):
  neural-head deferral and pivot to F-DATA-PACK; authoritative record for the
  decision to ship the curated dataset grid over a deep-learning competitor
- [ADR D40](docs/adr/D40-ia-aligned-training.md): IA-aligned training
  (palanca 1 sample weighting); palanca-1 verdict STOP, infrastructure stays
- [ADR D41](docs/adr/D41-universal-reranker.md): universal booster pipeline
  (pooled multi-manifest staging, temporal validation protocol, holdout-band
  evaluator, GOA self-prior reference)

## Tests

The lab ships a minimum pytest suite to guard the bits whose silent
breakage would invalidate the thesis Ch6 numbers. Run with::

    poetry install --quiet
    poetry run pytest -q tests/

The three pillars under `tests/`:

- `tests/test_schema_sha_determinism.py`: the 12-hex `schema_sha`
  derived from the feature column set must be order-independent and
  stable across dict and parquet roundtrips. A drift here silently
  invalidates every cached booster and breaks cross-PR Fmax
  comparability.
- `tests/test_golden_parquet_roundtrip.py`: load
  `tests/fixtures/golden_v27.parquet` (a tiny slice of
  `bench-v1-K5-v226-lineage-esm2_150m`), then assert reserved columns
  are present, every feature column lives in
  `protea_contracts.ALL_FEATURES`, numeric features are float, and
  labels are binary + finite. Skips with a clear hint if the fixture
  is absent.
- `tests/test_train_eval_split_determinism.py`: the train/val/eval
  split returned by `protea_reranker_lab.staging._decide_split` is
  reproducible across runs for both the `protein_group` (random
  k-fold) and `temporal` (snapshot-pair) strategies, including the
  negative-downsampling branch. Catches undocumented `random_state`
  defaults and `PYTHONHASHSEED` leaks.

CI runs the suite on Python 3.12 via `.github/workflows/test.yml`. One
pre-existing test that requires local-only `runs/transversal/*`
artefacts is deselected in CI; see the workflow comment for context.

## Linting and type checks

```bash
ruff check src scripts
mypy
poetry run pytest -q tests/
python scripts/check_smells.py --target src
poetry run sphinx-build -W -b html docs/source docs/_build/html
```

The CI reranker-token linter rejects bare reranker shorthand tokens
in prose. Use the axis-tuple form or a GOA snapshot name (v226, v230,
etc.) instead.

## Contributing

All changes go to `develop`; `main` tracks stable releases.

```bash
git checkout develop && git checkout -b feature/my-feature
pip install -e ".[dev]"
python scripts/run.py experiments/example.yaml  # smoke test
ruff check src scripts && mypy && poetry run pytest -q tests/
gh pr create -B develop --title "feat: ..." --body "..."
```

Key constraints:

- Dataset artefacts (`datasets/`, `runs/`) are git-ignored. Never
  commit large parquet files or trained model binaries.
- The feature schema is owned by `protea-contracts`; renaming features
  requires a coordinated PR pair and a contracts version bump.
- The lab consumes PROTEA artefacts; API changes belong in PROTEA.

## License

MIT. Copyright 2026 Francisco Miguel Pérez Canales. See `LICENSE`.
