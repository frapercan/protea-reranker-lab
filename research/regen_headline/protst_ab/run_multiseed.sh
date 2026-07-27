#!/usr/bin/env bash
# Multi-seed robustness driver for the ProtST reranker A/B.
# Reuses the enriched parquets (NO re-enrichment). For each new seed runs both
# arms (armA exclude-protst, armB include-protst), then scores the 9-cell.
# seed 42 is the already-existing armA_9cell.json / armB_9cell.json.
set -euo pipefail
cd /home/frapercan/Thesis2/storage/regen_headline/protst_ab
PY=/home/frapercan/Thesis2/repositories/protea-reranker-lab/.venv/bin/python
TRAIN=enriched_train.parquet
EVAL=enriched_eval.parquet

for SEED in 7 123 2024; do
  for ARM in armA armB; do
    OUT="out_seed${SEED}_${ARM}"
    echo "=== [$(date +%H:%M:%S)] TRAIN seed=${SEED} arm=${ARM} -> ${OUT} ==="
    "$PY" train_ab_seed.py "$TRAIN" "$EVAL" "$OUT" "$ARM" "$SEED" \
        > "${ARM}_seed${SEED}_train.log" 2>&1
    echo "=== [$(date +%H:%M:%S)] SCORE seed=${SEED} arm=${ARM} ==="
    "$PY" score_ab.py "${OUT}/eval_scores.parquet" "${ARM}_seed${SEED}_9cell.json" \
        > "${ARM}_seed${SEED}_score.log" 2>&1
    echo "=== [$(date +%H:%M:%S)] DONE seed=${SEED} arm=${ARM} ==="
  done
done
echo "=== ALL SEEDS DONE [$(date +%H:%M:%S)] ==="
