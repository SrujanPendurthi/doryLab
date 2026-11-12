#!/usr/bin/env bash
# Task 11: k=1 draft = APFS clone (no extra disk) of the stock GGUF with dory.num_loops=1 baked in.
set -euo pipefail
cd "$(dirname "$0")/.."
G=models/gguf/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf
D=models/gguf/dory-q5-loops1-draft.gguf
[ -f "$D" ] || cp -c "$G" "$D"
PYTHONPATH=vendor/llama.cpp/gguf-py .venv/bin/python vendor/llama.cpp/gguf-py/gguf/scripts/gguf_set_metadata.py --force "$D" dory.num_loops 1
