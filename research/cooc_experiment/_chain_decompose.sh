#!/bin/bash
# Wait for the P/S isolation to release its ~25GB, then run the order-vs-count decomposition.
while pgrep -f "isolate_percell_split.py" >/dev/null 2>&1; do sleep 30; done
echo "CHAIN: P/S finished, starting decomposition at $(date +%H:%M:%S)"
exec /home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python \
     /home/frapercan/Thesis2/storage/cooc_experiment/decompose_order_vs_count.py
