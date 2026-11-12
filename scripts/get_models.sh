#!/usr/bin/env bash
# Fetch Dory Q5_K_M GGUF (-> models/dory-q5.gguf) and BF16 metadata only (-> models/bf16-meta/).
set -euo pipefail
cd "$(dirname "$0")/.."
G=nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF
G_REV=df53755cb8741843af8e019e088819fbef17e29b
F=NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf
B=nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16
B_REV=f6c832e6963ddb74f62c75813a3e8c74b1af67b1
HF=${HF:-hf}
if [ ! -e models/dory-q5.gguf ]; then
  mkdir -p models
  free_kb=$(df -k models | awk 'NR==2{print $4}')
  if [ "$free_kb" -lt $((8*1024*1024)) ]; then
    echo "BLOCKED: need >= 8 GB free, have $((free_kb/1024/1024)) GB" >&2; exit 1
  fi
  $HF download $G $F --revision $G_REV --local-dir models/gguf
  ln -sf gguf/$F models/dory-q5.gguf
fi
$HF download $B config.json tokenizer_config.json "chat_template*" --revision $B_REV --local-dir models/bf16-meta
