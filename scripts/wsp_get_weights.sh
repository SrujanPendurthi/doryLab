#!/usr/bin/env bash
# Fetch the pinned Dory1.0-4B BF16 snapshot, link it as models/bf16, and build the Q8_0 / BF16 GGUFs.
set -euo pipefail
REV=f6c832e6963ddb74f62c75813a3e8c74b1af67b1
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
free_gb=$(df -g . | awk 'NR==2{print $4}')
[ "$free_gb" -ge 25 ] || { echo "free disk ${free_gb} GB < 25 GB, aborting" >&2; exit 1; }
snap=$($PY -c "from huggingface_hub import snapshot_download as d; print(d('nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16', revision='$REV'))")
mkdir -p models/gguf
# models/bf16 is a directory of symlinks into the HF cache (never copied, never deleted)
rm -rf models/bf16 && mkdir models/bf16
for f in "$snap"/*; do ln -s "$(readlink -f "$f")" "models/bf16/$(basename "$f")"; done
for t in q8_0 bf16; do
  out=models/gguf/dory-base-$t.gguf
  [ -f "$out" ] || $PY vendor/llama.cpp/convert_hf_to_gguf.py models/bf16 --outtype $t --outfile "$out"
done
