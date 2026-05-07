# protea-reranker-lab

Research sandbox for iterating on the PROTEA GO-term reranker without the full
PROTEA stack (Postgres / RabbitMQ / workers / API).

Core idea: PROTEA's expensive KNN feature computation is amortised by exporting
**frozen feature datasets** to parquet. The lab consumes the parquet + manifest,
fits LightGBM via a **streaming** PyArrow pipeline (no pandas materialisation),
evaluates per-cell, and publishes the winning booster back to PROTEA's
`RerankerModel` registry.

## Repo layout

```
protea-reranker-lab/
├── datasets/                  # frozen feature dumps (git-ignored, large)
│   └── bench-v1-K5/
│       ├── train.parquet      # 12 multisnap pairs v160→v220
│       ├── eval.parquet       # hold-out v220→v230, partitioned (cat, aspect)
│       ├── manifest.json      # schema sha, K, embedding, deltas
│       └── parent_map.json    # GO is_a/part_of edges (optional, propagation)
├── src/protea_reranker_lab/
│   ├── data.py                # PyArrow streaming primitives
│   ├── staging.py             # bucket-sort + cell filter + cat-encode + split
│   ├── sequences.py           # ParquetFeatureSequence (lgb.Sequence)
│   ├── reranker.py            # feature defs + streaming fit / predict
│   ├── evaluate.py            # numpy-only protein-grouped Fmax
│   ├── runner.py              # ExperimentSpec → run_experiment orchestrator
│   ├── bootstrap.py           # paired bootstrap CIs vs KNN baseline
│   ├── experiment.py          # ExperimentSpec / DatasetRef / SweepRef
│   ├── schemas.py             # ManifestV1 + DatasetSpec + schema_sha helpers
│   ├── builder.py             # streaming reshape (column prune + snap filter)
│   └── train.py               # CLI wrapper around run_experiment
├── scripts/
│   ├── run.py                 # run a single ExperimentSpec YAML
│   ├── build_study_specs.py   # generate the v9 study YAMLs (F1/F2/F4)
│   ├── run_study.py           # sequential, resumable phase orchestrator
│   ├── run_bootstrap_phase.py # F3 paired bootstrap driver
│   ├── summarise_study.py     # aggregate phase CSVs → SUMMARY.md
│   └── export_parent_map.py   # one-shot DB → parent_map.json (PROTEA venv)
└── runs/study_v9/             # study artefacts (git-ignored)
    ├── replication/           # F1: per-cell CSV + run.json + model.txt
    ├── ablation/              # F2: leave-one-family-out
    ├── hparam/                # F4: 3³ hparam grid on nk-bpo
    ├── bootstrap/             # F3: paired bootstrap CIs
    └── SUMMARY.md             # auto-generated, drop-in for thesis
```

## Quickstart

```bash
# 1. Install (Python 3.11+)
pip install -e .

# 2. Dataset must be present at datasets/bench-v1-K5/ (produced by PROTEA
#    via export_research_dataset). Verify manifest.json + train.parquet +
#    eval.parquet are there.

# 3. Run a single experiment from a YAML spec
python scripts/run.py experiments/_generated/study_v9/f1_replication/nk-bpo_seed42.yaml

# 4. Or run a whole study phase (sequential, resumable)
python scripts/build_study_specs.py     # generate 93 YAMLs
python scripts/run_study.py f1          # 27 specs replication
python scripts/run_study.py f2          # 39 specs ablation
python scripts/run_study.py f4          # 27 specs hparam grid

# 5. Bootstrap CIs vs KNN baseline (after f1 winners)
python scripts/run_bootstrap_phase.py

# 6. Aggregate everything into SUMMARY.md
python scripts/summarise_study.py
```

## Feature schema

52 features, identical layout to PROTEA's `reranker.py` (vendored constant).
Categorical encoding is performed at staging time using the lab's own code
maps; the resulting bucket parquets store int-encoded categoricals plus
float32 numerics.

## Streaming staging

`stage_for_training` produces sorted-by-protein bucket parquet files plus
`labels.npy` / `groups.npy` / `proteins.npy`. The trainer reads features
through `ParquetFeatureSequence` (an `lgb.Sequence` that lazy-loads row
groups) so RSS during fit stays bounded — peak ≈ 12 GB on the largest cell
(nk-bpo, 27.6M rows × 52 features).

Optional True-Path-Rule label propagation (`propagate_labels=True` +
`parent_map.json`) is **off by default** — the bench-v1-K5 dataset is
generated with PROTEA's reconciled-mode evaluation, which already applies
ancestor closure before producing `gt_pairs`. Re-propagating in the lab
causes double-propagation and degrades fmax.

<!-- protea-stack:start -->

## Repositories in the PROTEA stack

Single source of truth: [`docs/source/_data/stack.yaml`](https://github.com/frapercan/PROTEA/blob/develop/docs/source/_data/stack.yaml) in PROTEA. Run `python scripts/sync_stack.py` to regenerate this block.

| Repo | Role | Status | Summary |
|------|------|--------|---------|
| [PROTEA](https://github.com/frapercan/PROTEA) | Platform | `active` | Backend platform. Hosts the ORM, job queue, FastAPI surface, frontend, and orchestration. |
| [protea-contracts](https://github.com/frapercan/protea-contracts) | Contracts | `beta` | Shared contract surface. ABCs, pydantic payloads, feature schema, schema_sha. Imported by every other repo. |
| [protea-method](https://github.com/frapercan/protea-method) | Inference | `skeleton` | Pure inference path (KNN, feature compute, reranker apply). Target of the F2C extraction. Bind-mounted by the LAFA containers. |
| [protea-sources](https://github.com/frapercan/protea-sources) | Source plugin | `skeleton` | Annotation source plugins (GOA, QuickGO, UniProt). Discovered via Python entry_points. |
| [protea-runners](https://github.com/frapercan/protea-runners) | Runner plugin | `skeleton` | Experiment runner plugins (LightGBM lab, KNN baseline, future GNN). Discovered via Python entry_points. |
| [protea-backends](https://github.com/frapercan/protea-backends) | Backend plugin | `skeleton` | Protein language model embedding backends (ESM family, T5/ProstT5, Ankh, ESM3-C). Discovered via Python entry_points. |
| **protea-reranker-lab** (this repo) | Lab | `active` | LightGBM reranker training lab. Pulls datasets from PROTEA, trains boosters, publishes them back via /reranker-models/import-by-reference. |
| [cafaeval-protea](https://github.com/frapercan/cafaeval-protea) | Evaluator | `active` | Standalone fork of cafaeval (CAFA-evaluator-PK) with the PK-coverage fix and a bit-exact parity guarantee against the upstream. |

<!-- protea-stack:end -->
