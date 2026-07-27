#!/bin/bash
# SSE full-corpus auto-measure chain. Waits for training, then runs all stages to the receipt.
# Robust, detached, no self-killing monitors. Each stage logs to its own .out; status.json tracks progress.
set -u
D=/home/frapercan/Thesis2/storage/sse_full
PPY=/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python
LPY=/home/frapercan/Thesis2/repositories/protea-reranker-lab/.venv/bin/python
st(){ echo "{\"stage\":\"$1\",\"ts\":\"$(date -Iseconds)\"}" > $D/chain_status.json; }

# 1. wait for training sentinel (max ~2h)
st waiting_for_train
for i in $(seq 1 480); do
  grep -q "SSE FULL TRAIN DONE" $D/train.out 2>/dev/null && break
  sleep 15
done
grep -q "SSE FULL TRAIN DONE" $D/train.out 2>/dev/null || { st train_failed; exit 1; }

# 2. score (Mode A slices + Mode B covmin/covmean matrices)
st score
$PPY $D/sse_full_score.py > $D/score.out 2>&1 || { st score_failed; exit 1; }

# 3. Mode A generator measure (min consensus), PK+LK x 3 aspects
st modeA
$PPY $D/sse_full_modeA.py min PK,LK > $D/modeA.out 2>&1 || echo "modeA nonzero exit" >> $D/modeA.out

# 4. Mode B retrain (lab venv, lightgbm): baseline / sse / sse_shuffled
st modeB_retrain
$LPY $D/sse_full_modeB_retrain.py > $D/modeB_retrain.out 2>&1 || echo "retrain nonzero exit" >> $D/modeB_retrain.out

# 5. Mode B cafa + paired bootstrap CI, pk+lk x 3 aspects
st modeB_cafa
$PPY $D/sse_full_modeB_cafa.py pk,lk > $D/modeB_cafa.out 2>&1 || echo "cafa nonzero exit" >> $D/modeB_cafa.out

# 6. receipt
st receipt
$PPY $D/sse_full_receipt.py > $D/receipt.out 2>&1 || echo "receipt nonzero exit" >> $D/receipt.out
st done
echo "CHAIN DONE $(date -Iseconds)" >> $D/chain.out
