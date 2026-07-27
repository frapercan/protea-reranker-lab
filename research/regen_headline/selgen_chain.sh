#!/usr/bin/env bash
cd /home/frapercan/Thesis2/storage/regen_headline
PY=/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python
# wait for taxon extraction to finish
while ps aux | grep -q "[z]cat.*goa_uniprot"; do sleep 20; done
echo "=== taxon done, refreshing recon ===" 
$PY selgen_recon.py > selgen_recon_run.log 2>&1
echo "=== recon done, running main ==="
$PY selgen_main.py > selgen_main.log 2>&1
echo "=== ALL DONE ==="
