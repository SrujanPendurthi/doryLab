#!/usr/bin/env bash
# Follow-ups: lazy train top-up after regrade, Task 11 spec bench, router memgate (slots=4).
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY=.venv/bin/python
$PY -m dory.sweep --split train --ks 1,2,4 --lazy --variant q5 --slots 4 --kv f16 --out results/sweep_train.jsonl
echo "TOPUP_DONE $(date)"
$PY scripts/spec_bench.py
echo "SPEC_DONE $(date)"
$PY scripts/memgate.py 4
echo "FOLLOWUPS_DONE $(date)"
