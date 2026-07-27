#!/usr/bin/env bash
# Robust auto-measure chain for the clean entailment-over-kWTA experiment.
# Runs the measurement ITSELF (rescore + CI), then consolidates the receipt. Detached, self-contained.
set -u
cd /home/frapercan/Thesis2/storage/entail_kwta
PY=/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python
STATUS=chain_status.json
echo '{"stage":"measure","started":"'"$(date -Iseconds)"'"}' > "$STATUS"

echo "=== STEP 1: measure_convert_ci.py ==="
$PY measure_convert_ci.py > measure_convert_ci.log 2>&1
rc1=$?
if [ $rc1 -ne 0 ]; then
  echo '{"stage":"measure_FAILED","rc":'"$rc1"',"at":"'"$(date -Iseconds)"'"}' > "$STATUS"
  echo "MEASURE FAILED rc=$rc1"; tail -40 measure_convert_ci.log; exit $rc1
fi

echo '{"stage":"consolidate","at":"'"$(date -Iseconds)"'"}' > "$STATUS"
echo "=== STEP 2: consolidate.py ==="
$PY consolidate.py > consolidate.log 2>&1
rc2=$?
if [ $rc2 -ne 0 ]; then
  echo '{"stage":"consolidate_FAILED","rc":'"$rc2"',"at":"'"$(date -Iseconds)"'"}' > "$STATUS"
  echo "CONSOLIDATE FAILED rc=$rc2"; tail -40 consolidate.log; exit $rc2
fi

echo '{"stage":"done","at":"'"$(date -Iseconds)"'"}' > "$STATUS"
echo "CHAIN_DONE $(date -Iseconds)"
