#!/bin/bash
set -e
cd /home/frapercan/Thesis2/storage/sse_poc
PY=/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python
export SSE_NSUB=25000 SSE_SEEDS=5 SSE_EPOCHS=12 SSE_MINSUB=25 SSE_GAMMA=0.8
echo "{\"stage\":\"train\",\"ts\":$(date +%s)}" > chain_status.json
$PY sse_train_infer.py > train.out 2>&1
echo "{\"stage\":\"measure_min\",\"ts\":$(date +%s)}" > chain_status.json
$PY sse_measure.py min > measure_min.out 2>&1
echo "{\"stage\":\"measure_avg\",\"ts\":$(date +%s)}" > chain_status.json
$PY sse_measure.py avg > measure_avg.out 2>&1
echo "{\"stage\":\"done\",\"ts\":$(date +%s)}" > chain_status.json
