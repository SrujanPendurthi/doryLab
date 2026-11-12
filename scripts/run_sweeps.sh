#!/usr/bin/env bash
# Chained label sweeps (Tasks 5b/6b). Resumable: rerun to continue; transient errors retried.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
S="--variant q5 --slots 4 --kv f16"
for pass in 1 2 3; do
  $PY -m dory.sweep --split holdout --sources gsm8k,mmlu_pro --ks 1,2,4 $S --out results/sweep_holdout.jsonl
  $PY -m dory.sweep --split holdout --ks 1,2,4 $S --out results/sweep_holdout.jsonl
  $PY -m dory.sweep --split final   --ks 1,2,4 $S --out results/sweep_final.jsonl
  $PY -m dory.sweep --split train   --ks 1,2,4 --lazy $S --out results/sweep_train.jsonl
  echo "pass $pass done $(date)"
done
echo ALL_SWEEPS_DONE
