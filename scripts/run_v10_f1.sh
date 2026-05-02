#!/usr/bin/env bash
# Sequential v10 F1 launcher — invokes scripts/run.py per spec.
# Resumable: skips runs whose output dir already has a successful run.json.
set -uo pipefail
LAB=/home/frapercan/Thesis/repositories/protea-reranker-lab
PY="$LAB/.venv/bin/python"
SPECS_DIR="$LAB/experiments/_generated/study_v10/f1_replication"
LOG="$LAB/logs/study_v10_f1.log"
RESULTS="$LAB/runs/study_v10/replication/results.csv"
mkdir -p "$(dirname "$RESULTS")"

cd "$LAB"
echo "[$(date -Is)] === study_v10 f1 sequential launch ===" | tee -a "$LOG"
echo "spec,phase,status,exit_code,fmax,best_iteration,n_train,n_val,n_eval,duration_s,output_dir" > "$RESULTS"

i=0
total=$(ls "$SPECS_DIR"/*.yaml | wc -l)
for spec in "$SPECS_DIR"/*.yaml; do
  i=$((i+1))
  name=$(basename "$spec" .yaml)
  out_dir=$($PY -c "import yaml; print(yaml.safe_load(open('$spec'))['output_dir'])")
  run_json="$LAB/$out_dir/run.json"

  if [ -f "$run_json" ] && $PY -c "import json; assert json.load(open('$run_json')).get('status')=='ok'" 2>/dev/null; then
    echo "[skip $i/$total] $name" | tee -a "$LOG"
    continue
  fi

  rm -rf "$LAB/$out_dir" 2>/dev/null
  echo "[$(date -Is)] [$i/$total] $name …" | tee -a "$LOG"
  t0=$(date +%s)
  "$PY" "$LAB/scripts/run.py" "$spec" >> "$LOG" 2>&1
  rc=$?
  dur=$(($(date +%s) - t0))

  if [ $rc -eq 0 ] && $PY -c "import json; assert json.load(open('$run_json')).get('status')=='ok'" 2>/dev/null; then
    fmax=$($PY -c "import json; d=json.load(open('$run_json'))['metrics']; print(d.get('test_fmax',''))")
    bi=$($PY -c "import json; d=json.load(open('$run_json'))['metrics']; print(d.get('best_iteration',''))")
    nt=$($PY -c "import json; d=json.load(open('$run_json')).get('split',{}); print(d.get('n_train',''))")
    nv=$($PY -c "import json; d=json.load(open('$run_json')).get('split',{}); print(d.get('n_val',''))")
    ne=$($PY -c "import json; d=json.load(open('$run_json')).get('split',{}); print(d.get('n_eval',''))")
    echo "$name,f1,ok,0,$fmax,$bi,$nt,$nv,$ne,$dur,$out_dir" >> "$RESULTS"
    echo "[done $i/$total] $name fmax=$fmax dur=${dur}s" | tee -a "$LOG"
  else
    echo "$name,f1,failed,$rc,,,,,,$dur,$out_dir" >> "$RESULTS"
    echo "[FAIL $i/$total] $name exit=$rc" | tee -a "$LOG"
  fi
done

echo "[$(date -Is)] === study_v10 f1 done ===" | tee -a "$LOG"
