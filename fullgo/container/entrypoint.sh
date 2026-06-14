#!/usr/bin/env bash
# LAFA container contract: --query_file -q, --train_sequences, --annot_file -a,
# --graph, --output_file -o. Reference bundle bind-mounted at /app/data.
set -euo pipefail
exec python3 /app/predict.py "$@"
