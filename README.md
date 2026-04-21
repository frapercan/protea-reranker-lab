# protea-reranker-lab

Research sandbox for iterating on the PROTEA GO-term reranker without the full
PROTEA stack (Postgres / RabbitMQ / workers / API).

Core idea: PROTEA's expensive KNN feature computation is amortised by dumping
**frozen feature datasets** to parquet. Experiments load parquet → fit LightGBM →
evaluate → log to Weights & Biases. Iteration cycle drops from hours to minutes.

## Repo layout

```
protea-reranker-lab/
├── datasets/                  # frozen feature dumps (git-ignored, large)
│   └── bench-v1-K5/
│       ├── train.parquet      # (protein, go, label, cat, aspect, snapshot) + features
│       ├── eval.parquet       # hold-out v220→v230, partitioned (cat, aspect)
│       └── manifest.json      # schema sha, K, embedding, deltas
├── src/protea_reranker_lab/
│   ├── data.py                # load parquet → (X, y, group_ids)
│   ├── reranker.py            # feature defs + LightGBM fit (mirrors PROTEA)
│   ├── evaluate.py            # per-cell CAFA fmax
│   └── train.py               # wandb-instrumented training entrypoint
├── scripts/
│   ├── dump_dataset.py        # PROTEA DB → parquet (requires PROTEA repo + DB access)
│   └── upload_model.py        # push winning model → PROTEA reranker_model
├── sweeps/
│   ├── baseline.yaml          # Bayesian sweep over LightGBM hparams
│   └── feature_ablation.yaml  # grid over feature-family drops
└── experiments/               # YAML configs + logs (committed)
```

## Status

Scaffold + PROTEA dump hook ready. The heavy path (KNN + all feature families,
per-cell × multisnap) is produced by the `train_reranker_auto` operation in
PROTEA with `dump_only=True`, which writes the frozen `train.parquet` +
`eval.parquet` + `manifest.json` directly into `datasets/`.

The alternate `scripts/dump_dataset.py` in this repo is still useful when you
already have a ``prediction_set`` + ``evaluation_set`` in the DB and only want
the eval side (no multisnap history).

## Quickstart

```bash
# 1. Install (Python 3.11+)
pip install -e .

# 2a. Full dump (train+eval, multisnap, from PROTEA)
#     Run this from the PROTEA repo with its worker stack up:
cd ~/Thesis/repositories/PROTEA
python scripts/dump_reranker_dataset.py \
    --name bench-v1-K5 \
    --out ~/Thesis/repositories/protea-reranker-lab/datasets/bench-v1-K5 \
    --train-versions 160 165 170 175 180 185 190 195 200 205 211 215 220 \
    --test-versions 230 \
    --k 5 --all-features

# 2b. Eval-only dump (from existing prediction_set + evaluation_set)
python scripts/dump_dataset.py \
    --prediction-set 4b734d30-29b3-48ce-a1f7-e9cf3b57156d \
    --evaluation-set a73cb77c-9adf-4d55-b61f-c0b1bd05be01 \
    --out datasets/bench-v1-K5/eval.parquet

# 3. Run a single training (no W&B)
python -m protea_reranker_lab.train \
    --dataset datasets/bench-v1-K5 \
    --cell pk-bpo \
    --num-boost-round 3000

# 4. Launch a W&B sweep
wandb sweep sweeps/baseline.yaml   # prints SWEEP_ID
wandb agent SWEEP_ID               # agent in any shell (can run N in parallel)
```

## Feature schema

52 features, identical layout to PROTEA's `reranker.py` (vendored constant).
Categorical encoding is deferred to load time (parquet stores raw strings).
