#!/usr/bin/env bash
# Task 6 Step 1: holdout mmlu_pro (100 items) at k=1,2,4 with max_tokens 1024. Resumable.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; [ -x $PY ] || PY=/Users/srujanpendurthi/Projects/Current/doryAgent/.venv/bin/python
PYTHONPATH=. $PY -m dory.sweep --split holdout --sources mmlu_pro --max-tokens 1024 --ks 1,2,4 \
  --variant mmlu1024 --slots 4 --kv f16 --out results/sweep_holdout_mmlu1024.jsonl
