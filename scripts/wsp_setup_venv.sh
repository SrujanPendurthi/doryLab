#!/usr/bin/env bash
# venv with a transformers build that knows `dory` (PR #50099 head), needed by the GGUF converter.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv --system-site-packages .venv-wsp
.venv-wsp/bin/pip install "git+https://github.com/huggingface/transformers.git@fcc6924dfbac964f00787859b568c867ef5ad4cb"
