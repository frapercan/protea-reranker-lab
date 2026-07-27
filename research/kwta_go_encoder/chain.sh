#!/bin/bash
# Waits for Phase 1 to finish, then runs intrinsic report, Phase 2 (learned + fixed two-tower
# control), separability, and the f_micro_w decider. Sequential; each stage logs to its own file.
set -e
cd /home/frapercan/Thesis2
PY=repositories/PROTEA/.venv/bin/python
D=storage/kwta_go_encoder
echo "waiting for Phase 1..."
while ! grep -q "DONE ablation" "$D/train_p1.log" 2>/dev/null; do sleep 20; done
echo "Phase 1 done: $(date)"

$PY $D/intrinsic_report.py            > $D/intrinsic.log 2>&1
echo "intrinsic done: $(date)"

$PY $D/train_seq_projection.py coann_struct_text > $D/p2_learned.log 2>&1
echo "p2 learned done: $(date)"
$PY $D/train_seq_projection.py twotower           > $D/p2_twotower.log 2>&1
echo "p2 twotower done: $(date)"

$PY $D/evaluate_separability.py coann_struct_text > $D/sep_learned.log 2>&1
echo "sep learned done: $(date)"
$PY $D/evaluate_separability.py twotower          > $D/sep_twotower.log 2>&1
echo "sep twotower done: $(date)"

$PY $D/evaluate_fmicrow.py coann_struct_text      > $D/fmicrow.log 2>&1
echo "fmicrow done: $(date)"
echo "CHAIN COMPLETE: $(date)"
