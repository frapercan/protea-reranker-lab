#!/bin/bash
while pgrep -f "anchor_deployed_recipe.py" >/dev/null 2>&1; do sleep 20; done
echo "CHAIN: anchor done, starting cross-aspect ceiling at $(date +%H:%M:%S)"
exec /home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python \
     /home/frapercan/Thesis2/storage/cooc_experiment/cross_aspect_channel_ceiling.py
