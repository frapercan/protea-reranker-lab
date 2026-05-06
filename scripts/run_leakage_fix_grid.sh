#!/usr/bin/env bash
# Regenerate the 27 v9 replication runs against the filtered dataset.
# Source specs live in runs/study_v9/replication/<cell>_seed<N>/spec.yaml.
# We rewrite ``dataset.manifest`` and ``output_dir`` and run sequentially.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=.venv/bin/python
SRC_ROOT=runs/study_v9/replication
DST_ROOT=runs/leakage_fix/grid
SPEC_ROOT=experiments/leakage_fix/grid
mkdir -p "$SPEC_ROOT" "$DST_ROOT"

for cell in nk-bpo nk-cco nk-mfo lk-bpo lk-cco lk-mfo pk-bpo pk-cco pk-mfo; do
  for seed in 7 42 137; do
    name="${cell}_seed${seed}"
    src="$SRC_ROOT/$name/spec.yaml"
    spec="$SPEC_ROOT/$name.yaml"
    [[ -f "$src" ]] || { echo "skip missing $src"; continue; }
    sed -e 's|datasets/bench-v1-K5/manifest.json|datasets/bench-v1-K5-filtered/manifest.json|' \
        -e "s|^name: .*|name: leakfix_${name}|" \
        -e "s|^output_dir: .*|output_dir: ${DST_ROOT}/${name}|" \
        "$src" > "$spec"
    if [[ -f "$DST_ROOT/$name/run.json" ]] && \
       grep -q '"status": "ok"' "$DST_ROOT/$name/run.json" 2>/dev/null; then
      echo "skip done $name"
      continue
    fi
    echo "=== $name ==="
    "$PYTHON" -u scripts/run.py "$spec" 2>&1 | tail -3
  done
done

echo
echo "=== summary ==="
for cell in nk-bpo nk-cco nk-mfo lk-bpo lk-cco lk-mfo pk-bpo pk-cco pk-mfo; do
  for seed in 7 42 137; do
    rj="$DST_ROOT/${cell}_seed${seed}/run.json"
    [[ -f "$rj" ]] || continue
    fmax=$("$PYTHON" -c "import json; print(f'{json.load(open(\"$rj\"))[\"metrics\"][\"test_fmax\"]:.4f}')")
    printf "  %-18s seed=%-3s test_fmax=%s\n" "$cell" "$seed" "$fmax"
  done
done
