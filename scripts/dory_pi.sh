#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if pgrep -x llama-server >/dev/null; then
    echo "Cannot start Dory: a llama-server is already running." >&2
    exit 1
fi
if lsof -nP -iTCP:8083 -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Cannot start Dory: port 8083 is already in use." >&2
    exit 1
fi

# vendor is a symlink to the main repo in a worktree; pwd -P resolves it.
export DORY_LLAMA_BIN="$(cd "$ROOT/vendor" && pwd -P)/llama.cpp-loops/build/bin"
exec "$ROOT/.venv/bin/python" -m dory.gateway --ctx 24576
