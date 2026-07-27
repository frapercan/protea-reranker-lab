#!/usr/bin/env bash
# Detached auto-chain: train frozen-rep SSE (K=5) -> score + measure generalization -> receipt.
set -u
cd /home/frapercan/Thesis2/storage/sse_kwta_gen
VP=/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python
ST=chain_status.json
echo '{"stage":"train","ts":"'$(date -Is)'"}' > $ST
export SSE_SEEDS=5 SSE_EPOCHS=12 SSE_MINCOUNT=25
$VP build_and_train.py > train.out 2>&1
RC=$?
if [ $RC -ne 0 ]; then echo '{"stage":"train_FAILED","rc":'$RC',"ts":"'$(date -Is)'"}' > $ST; exit 1; fi
echo '{"stage":"measure","ts":"'$(date -Is)'"}' > $ST
$VP score_and_measure.py > measure.out 2>&1
RC=$?
if [ $RC -ne 0 ]; then echo '{"stage":"measure_FAILED","rc":'$RC',"ts":"'$(date -Is)'"}' > $ST; exit 1; fi
echo '{"stage":"done","ts":"'$(date -Is)'"}' > $ST
