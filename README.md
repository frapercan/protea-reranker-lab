# protea-reranker-lab

Offline LightGBM training laboratory for the PROTEA GO-term reranker.
The lab consumes frozen feature datasets exported from PROTEA, fits a
streaming ranking model without ever loading a full DataFrame into
memory, evaluates per ontology-aspect cell, and publishes winning
boosters back to PROTEA's `RerankerModel` registry.

<!-- protea-stack:start -->

## Repositories in the PROTEA stack

Single source of truth: [`docs/source/_data/stack.yaml`](https://github.com/frapercan/PROTEA/blob/develop/docs/source/_data/stack.yaml) in PROTEA. Run `python scripts/sync_stack.py` to regenerate this block.

| Repo | Role | Status | Summary |
|------|------|--------|---------|
| [PROTEA](https://github.com/frapercan/PROTEA) | Platform | `active` | Backend platform. Hosts the ORM, job queue, FastAPI surface, frontend, and orchestration. |
| [protea-contracts](https://github.com/frapercan/protea-contracts) | Contracts | `beta` | Shared contract surface. ABCs, pydantic payloads, feature schema, schema_sha. Imported by every other repo. |
| [protea-method](https://github.com/frapercan/protea-method) | Inference | `active` | Pure inference path (KNN, feature compute, reranker apply). LAFA inference layer; publishes to DockerHub. |
| [protea-sources](https://github.com/frapercan/protea-sources) | Source plugin | `skeleton` | Annotation source plugins (GOA, QuickGO, UniProt). Discovered via Python entry_points. |
| [protea-runners](https://github.com/frapercan/protea-runners) | Runner plugin | `skeleton` | Experiment runner plugins (LightGBM lab, KNN baseline, future GNN). Discovered via Python entry_points. |
| [protea-backends](https://github.com/frapercan/protea-backends) | Backend plugin | `skeleton` | Protein language model embedding backends (ESM family, T5/ProstT5, Ankh, ESM3-C). Discovered via Python entry_points. |
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

**Current champion (v27-binary, multi-seed, 2026-05-18):**
NK+LK selective average cafaeval Fmax **0.7291 +/- 0.0028** on
`bench-v1-K5-v226-lineage` (3 seeds). All six NK+LK paired-bootstrap
confidence intervals vs the KNN baseline are strictly positive at 95%
(N=10000). Selective deployment policy: NK+LK cells deploy v27-binary;
PK cells remain on the KNN baseline (PK gains are policy-zero; see ADR
D34). This is the publishable number for Chapter 6 of the doctoral thesis.

The earlier LB.2 estimate (0.6215 +/- 0.0014) is superseded by v27-binary
and should not be cited in place of 0.7291 in new writing. See the
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

## Install

Python 3.11 or later is required.

```bash
git clone https://github.com/frapercan/protea-reranker-lab.git
cd protea-reranker-lab
git checkout develop

pip install -e .          # runtime only
pip install -e ".[dev]"   # + ruff / mypy / sphinx
```

## Dataset pull, train, evaluate, import flow

### Step 1 — pull dataset from PROTEA

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

### Step 2 — train

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

### Step 3 — evaluate

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

### Step 4 — compare vs KNN baseline

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

### Step 5 — import winner to PROTEA

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
**0.6215 +/- 0.0014** LB.2 estimate. A subsequent v27-binary training
run (three seeds, binary classification objective instead of lambdarank)
produced the current champion **0.7291 +/- 0.0028**. The progression is:
0.4562 (artefact, do not cite) to 0.6215 (LB.2, superseded) to 0.7291
(v27-binary, current; cite this for all new writing on bench-v1-K5-v226-lineage NK+LK).

The selective-rerank decision (ADR D34) is documented in
`docs/decisions/D34-selective-rerank-policy.md` and the formal run
records are in `experiments/study-selective-rerank-K10-v226/`.

## Repo layout

```
protea-reranker-lab/
├── datasets/                  # frozen feature dumps (git-ignored)
├── src/protea_reranker_lab/
│   ├── train.py       # CLI entry-point
│   ├── evaluate.py    # numpy Fmax
│   ├── compare.py     # bootstrap comparison namespace
│   ├── staging.py     # bucket-sort + cell filter
│   ├── bootstrap.py   # paired bootstrap CIs
│   ├── runner.py      # ExperimentSpec orchestrator
│   ├── reranker.py    # LightGBM fit / predict
│   ├── sequences.py   # lgb.Sequence on parquet buckets
│   ├── builder.py     # streaming reshape
│   ├── data.py        # PyArrow streaming primitives
│   ├── experiment.py  # ExperimentSpec / DatasetRef
│   └── schemas.py     # ManifestV1 + schema_sha
├── scripts/           # CLI drivers (run.py, run_study.py, ...)
├── experiments/       # YAML spec files
└── docs/              # Sphinx documentation
```

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
mypy src
cd docs && make html
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
ruff check src scripts && mypy src && cd docs && make html
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
