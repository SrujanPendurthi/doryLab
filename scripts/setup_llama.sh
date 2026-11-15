#!/usr/bin/env bash
# Build the Dory fork with Metal. --loops builds only the runtime-loops variant.
set -euo pipefail
LLAMA_REF=1c3cdd23787750814aa21a070dc4d6dc451af871   # head of ggml-org/llama.cpp PR #30306
cd "$(dirname "$0")/.."
ROOT=$PWD
case "${1:-}" in
    ""|--loops) ;;
    *) echo "usage: $0 [--loops]" >&2; exit 2 ;;
esac
[ -d vendor/llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp vendor/llama.cpp
if [ "${1:-}" = "--loops" ]; then
    # Leave the stock checkout and its build untouched.
    git -C vendor/llama.cpp cat-file -e "$LLAMA_REF"
    [ -d vendor/llama.cpp-loops ] || git -C vendor/llama.cpp worktree add -b dory-runtime-loops ../llama.cpp-loops "$LLAMA_REF"
    cd vendor/llama.cpp-loops
    PATCH=$ROOT/patches/llama-dory-runtime-loops.patch
    if ! git apply --reverse --check "$PATCH" 2>/dev/null; then
        git apply --check "$PATCH"
        git apply "$PATCH"
    fi
else
    cd vendor/llama.cpp
    git cat-file -e "$LLAMA_REF" 2>/dev/null || git fetch --depth 1 origin pull/30306/head:dory
    git checkout "$LLAMA_REF"
fi
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 10 --target llama-server llama-cli llama-completion llama-lookup llama-quantize
build/bin/llama-server --version
