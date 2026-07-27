#!/bin/bash
# Evidence-aware SSE auto-chain: waits for the GAF parse, builds evidence labels + wall diagnostic (CPU),
# then GPU-polls (never kills) and trains/scores the 3 de-circularisation arms, measures, and writes the
# receipt. Detached; status in chain_status.json; each stage logs to its own .out.
set -u
D=/home/frapercan/Thesis2/storage/sse_evidence
PPY=/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python
st(){ echo "{\"stage\":\"$1\",\"ts\":\"$(date -Iseconds)\"}" > $D/chain_status.json; echo "=== STAGE $1 $(date -Iseconds) ===" >> $D/chain.out; }
gpu_wait(){
  while true; do
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if [ "${f:-0}" -gt 8000 ]; then echo "GPU free ${f}MiB -> go $(date -Iseconds)" >> $D/chain.out; break; fi
    echo "GPU free ${f}MiB <=8000, wait $(date -Iseconds)" >> $D/chain.out; sleep 30
  done
}

# 1. wait for GAF parse (max ~3h)
st waiting_for_gaf
for i in $(seq 1 720); do [ -f $D/gaf_parse.done ] && break; sleep 15; done
[ -f $D/gaf_parse.done ] || { st gaf_failed; exit 1; }

# 2. build evidence labels + calib features + term profiles (CPU)
st build_labels
$PPY $D/build_evidence_labels.py > $D/build_labels.out 2>&1 || { st build_failed; exit 1; }

# 3. wall-by-evidence diagnostic (CPU, independent of GPU arms)
st wall
$PPY $D/wall_by_evidence.py > $D/wall.out 2>&1 || echo "wall nonzero exit" >> $D/wall.out

# 4. train + score the 3 arms (GPU)
for arm in full noiea exp; do
  st train_$arm; gpu_wait
  $PPY $D/sse_ev_train.py $arm > $D/train_$arm.out 2>&1 || echo "train $arm nonzero" >> $D/train_$arm.out
  st score_$arm; gpu_wait
  $PPY $D/sse_ev_score.py $arm > $D/score_$arm.out 2>&1 || echo "score $arm nonzero" >> $D/score_$arm.out
done

# 5. measure (ranking AUC + calibrated wall checks + arms b/c); CPU + cafa subprocess
st measure
$PPY $D/sse_ev_measure.py > $D/measure.out 2>&1 || echo "measure nonzero" >> $D/measure.out

# 6. receipt
st receipt
$PPY $D/sse_ev_receipt.py > $D/receipt.out 2>&1 || echo "receipt nonzero" >> $D/receipt.out

st done
echo "CHAIN DONE $(date -Iseconds)" >> $D/chain.out
