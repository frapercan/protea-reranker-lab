#!/usr/bin/env bash
# FARM-EXP.9b: sequential, resumable runner for the 69 remaining cells.
#
# Iterates all spec YAMLs in experiments/farm_exp_9/ that correspond to
# pending entries in cells_to_rerun.csv, skipping any whose run.json
# already reports status=ok (idempotent / --resume semantics are built-in).
#
# Heartbeats are emitted to the agent-farm sqlite after each cell so the
# dashboard shows live progress.
#
# Usage:
#   bash scripts/farm_exp_9b_run_all.sh
#   # or with custom heartbeat task id:
#   TASK_ID=my-task-id bash scripts/farm_exp_9b_run_all.sh
#
# Notes:
#   - NOT comparable to pre-leakage bench-v1-K5 results (uses bench-v1-K5-filtered).
#   - Run order: replication > ablation > hparam > standalone (priority order).
#   - Estimated wall-clock: 70-100 h on a 12-CPU machine (~60-90 min/cell).
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=.venv/bin/python
SPEC_DIR=experiments/farm_exp_9
AGENT_FARM_ROOT="${AGENT_FARM_ROOT:-$HOME/Thesis2/agent-farm}"
DB_PY="$AGENT_FARM_ROOT/scripts/lib/db.py"
TASK_ID="${TASK_ID:-farm-exp-9b-runner-detached}"
LOG_TS=$(date +%Y%m%dT%H%M%S)

_heartbeat() {
    local level="$1"
    local msg="$2"
    if [[ -f "$DB_PY" ]]; then
        python3 "$DB_PY" heartbeat "$TASK_ID" "$level" "$msg" 2>/dev/null || true
    fi
}

_is_done() {
    local spec_yaml="$1"
    local out_dir
    out_dir=$(grep "^output_dir:" "$spec_yaml" | awk '{print $2}')
    local run_json="$out_dir/run.json"
    if [[ -f "$run_json" ]] && grep -q '"status": "ok"' "$run_json" 2>/dev/null; then
        return 0
    fi
    return 1
}

# Build ordered list of spec YAMLs: rep > abl > hp > standalone
SPECS=()
for f in "$SPEC_DIR"/rep_*.yaml; do
    [[ -f "$f" ]] && SPECS+=("$f")
done
for f in "$SPEC_DIR"/abl_*.yaml; do
    [[ -f "$f" ]] && SPECS+=("$f")
done
for f in "$SPEC_DIR"/hp_*.yaml; do
    [[ -f "$f" ]] && SPECS+=("$f")
done
for f in "$SPEC_DIR"/standalone_*.yaml; do
    [[ -f "$f" ]] && SPECS+=("$f")
done

TOTAL=${#SPECS[@]}
DONE_COUNT=0
SKIP_COUNT=0
FAIL_COUNT=0

echo "=== FARM-EXP.9b run_all @ $LOG_TS ==="
echo "Total specs: $TOTAL"
echo "TASK_ID: $TASK_ID"
echo ""

_heartbeat "info" "FARM-EXP.9b starting: $TOTAL specs, TASK_ID=$TASK_ID"

N=0
for spec in "${SPECS[@]}"; do
    N=$((N + 1))
    name=$(basename "$spec" .yaml)

    if _is_done "$spec"; then
        echo "[$N/$TOTAL] SKIP (already ok): $name"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        continue
    fi

    echo "[$N/$TOTAL] RUN: $name"
    _heartbeat "info" "cell $N/$TOTAL starting: $name"

    if "$PYTHON" -u scripts/run.py "$spec" 2>&1; then
        DONE_COUNT=$((DONE_COUNT + 1))
        echo "[$N/$TOTAL] OK: $name"
        _heartbeat "info" "cell $N/$TOTAL done: $name (ok)"
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "[$N/$TOTAL] FAILED: $name (continuing)"
        _heartbeat "warning" "cell $N/$TOTAL failed: $name (continuing)"
    fi
done

echo ""
echo "=== FARM-EXP.9b complete ==="
echo "Total: $TOTAL  Done: $DONE_COUNT  Skipped: $SKIP_COUNT  Failed: $FAIL_COUNT"
_heartbeat "info" "FARM-EXP.9b finished: total=$TOTAL done=$DONE_COUNT skipped=$SKIP_COUNT failed=$FAIL_COUNT"

if [[ $FAIL_COUNT -gt 0 ]]; then
    echo "Some cells failed. Check run.json files under runs/transversal/farm_exp_9_*/"
    exit 1
fi
exit 0
