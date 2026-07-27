#!/bin/bash
# Wait for the eval-abstract fetch to finish, then run the does-it-pay semantic scorer.
cd /home/frapercan/Thesis2
VENV=repositories/protea-reranker-lab/.venv/bin/python
FETCHLOG=storage/cooc_experiment/fetch_eval_abstracts.log
SCORELOG=storage/cooc_experiment/score_the_extras_semantic.log
echo "[orchestrator] waiting for fetch DONE..." > "$SCORELOG"
while ! grep -q "^DONE" "$FETCHLOG" 2>/dev/null; do
  sleep 30
  # bail if fetch process died without DONE (no python matching the fetch script)
  if ! pgrep -f fetch_eval_abstracts.py >/dev/null 2>&1 && ! grep -q "^DONE" "$FETCHLOG"; then
    echo "[orchestrator] fetch process gone without DONE; proceeding anyway (cache is resumable)" >> "$SCORELOG"
    break
  fi
done
echo "[orchestrator] fetch complete, launching scorer $(date -u +%H:%M:%S)" >> "$SCORELOG"
$VENV storage/cooc_experiment/score_the_extras_semantic.py >> "$SCORELOG" 2>&1
echo "[orchestrator] scorer exited $?" >> "$SCORELOG"
