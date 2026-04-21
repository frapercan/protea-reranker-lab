#!/usr/bin/env bash
# bench-v1-K5 dump + auto-launch pk-bpo sweep.
# All output lands in ~/Thesis/overnight.log for after-the-fact inspection.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
POETRY=/home/frapercan/.local/bin/poetry

LAB=/home/frapercan/Thesis/repositories/protea-reranker-lab
PROTEA=/home/frapercan/Thesis/repositories/PROTEA
LOG=/home/frapercan/Thesis/overnight.log
DATASET_DIR=$LAB/datasets/bench-v1-K5

echo "[$(date -Is)] === overnight run start ===" | tee -a "$LOG"
echo "[$(date -Is)] submitting dump bench-v1-K5 (13 pairs, --all-features)" | tee -a "$LOG"

cd "$PROTEA"
"$POETRY" run python scripts/dump_reranker_dataset.py \
  --name bench-v1-K5 \
  --out "$DATASET_DIR" \
  --train-versions 160 165 170 175 180 185 190 195 200 205 211 215 220 \
  --test-versions 230 \
  --k 5 --all-features 2>&1 | tee -a "$LOG"

if [[ ! -f "$DATASET_DIR/manifest.json" ]]; then
  echo "[$(date -Is)] dump failed — no manifest.json, aborting sweep" | tee -a "$LOG"
  exit 1
fi

echo "[$(date -Is)] dump OK — launching pk-bpo W&B sweep" | tee -a "$LOG"
cd "$LAB"
SWEEP_CREATE=$(.venv/bin/wandb sweep sweeps/bench_pk_bpo.yaml 2>&1)
echo "$SWEEP_CREATE" | tee -a "$LOG"
SWEEP_PATH=$(echo "$SWEEP_CREATE" | grep -oE "frapercan/protea-reranker-bench/[a-z0-9]+" | tail -1)

if [[ -z "$SWEEP_PATH" ]]; then
  echo "[$(date -Is)] could not extract sweep id — aborting" | tee -a "$LOG"
  exit 1
fi

echo "[$(date -Is)] sweep=$SWEEP_PATH, starting agent (count=40)" | tee -a "$LOG"
.venv/bin/wandb agent --count 40 "$SWEEP_PATH" 2>&1 | tee -a "$LOG"
echo "[$(date -Is)] === overnight run done ===" | tee -a "$LOG"
