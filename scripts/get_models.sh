#!/usr/bin/env bash
# Fetch Dory Q5_K_M GGUF (-> models/dory-q5.gguf) and BF16 metadata only (-> models/bf16-meta/).
set -euo pipefail
cd "$(dirname "$0")/.."
G=nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF
F=NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf
B=nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16
HF=${HF:-hf}
$HF download $G $F --local-dir models/gguf
ln -sf gguf/$F models/dory-q5.gguf
$HF download $B config.json tokenizer_config.json "chat_template*" --local-dir models/bf16-meta
