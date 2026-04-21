#!/usr/bin/env bash
# Waits for bench-v1-K5 manifest.json, then launches pk-bpo W&B sweep agent.
# Runs detached; log goes to ~/Thesis/overnight.log.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

LAB=/home/frapercan/Thesis/repositories/protea-reranker-lab
LOG=/home/frapercan/Thesis/overnight.log
MANIFEST=$LAB/datasets/bench-v1-K5/manifest.json
JOB_ID=e935f46a-4da4-4ff5-b420-14aed70e7e77

echo "[$(date -Is)] wait_and_sweep: polling for manifest" >> "$LOG"

# Poll job status every 60s. Bail on failure/cancel. Arm sweep on manifest.
while true; do
  status=$(curl -s -m 10 "http://127.0.0.1:8000/jobs/$JOB_ID" \
             | python3 -c 'import json,sys; print(json.load(sys.stdin).get("status",""))' 2>/dev/null || true)
  echo "[$(date -Is)] job status=$status" >> "$LOG"
  if [[ "$status" == "succeeded" ]]; then
    echo "[$(date -Is)] job succeeded" >> "$LOG"
    break
  fi
  if [[ "$status" == "failed" || "$status" == "cancelled" ]]; then
    echo "[$(date -Is)] job $status — aborting sweep" >> "$LOG"
    exit 1
  fi
  sleep 60
done

if [[ ! -f "$MANIFEST" ]]; then
  echo "[$(date -Is)] manifest missing after succeeded — aborting" >> "$LOG"
  exit 1
fi

echo "[$(date -Is)] manifest present — launching W&B sweep" >> "$LOG"
cd "$LAB"
SWEEP_CREATE=$(.venv/bin/wandb sweep sweeps/bench_pk_bpo.yaml 2>&1)
echo "$SWEEP_CREATE" >> "$LOG"
SWEEP_PATH=$(echo "$SWEEP_CREATE" | grep -oE "frapercan/protea-reranker-bench/[a-z0-9]+" | tail -1)
if [[ -z "$SWEEP_PATH" ]]; then
  echo "[$(date -Is)] could not parse sweep id" >> "$LOG"
  exit 1
fi
echo "[$(date -Is)] sweep=$SWEEP_PATH starting agent (count=40)" >> "$LOG"
.venv/bin/wandb agent --count 40 "$SWEEP_PATH" >> "$LOG" 2>&1
echo "[$(date -Is)] === sweep agent exit ===" >> "$LOG"
