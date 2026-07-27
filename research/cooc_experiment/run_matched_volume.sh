#!/bin/bash
cd /home/frapercan/Thesis2
VENV=repositories/protea-reranker-lab/.venv/bin/python
LOG=storage/cooc_experiment/union_matched_volume.log
echo "[chain] waiting for first scorer to exit..." > "$LOG"
while pgrep -f score_the_extras_semantic.py >/dev/null 2>&1; do sleep 20; done
echo "[chain] first scorer done, launching matched-volume $(date -u +%H:%M:%S)" >> "$LOG"
$VENV storage/cooc_experiment/union_matched_volume.py >> "$LOG" 2>&1
echo "[chain] matched-volume exited $?" >> "$LOG"
