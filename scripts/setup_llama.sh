#!/usr/bin/env bash
# Build the llama.cpp Dory fork (PR #30306 head) with Metal into vendor/llama.cpp/build.
set -euo pipefail
LLAMA_REF=1c3cdd23787750814aa21a070dc4d6dc451af871   # head of ggml-org/llama.cpp PR #30306
cd "$(dirname "$0")/.."
[ -d vendor/llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp vendor/llama.cpp
cd vendor/llama.cpp
git cat-file -e "$LLAMA_REF" 2>/dev/null || git fetch --depth 1 origin pull/30306/head:dory
git checkout "$LLAMA_REF"
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 10 --target llama-server llama-cli llama-completion llama-lookup
build/bin/llama-server --version
