import json
import os
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BIN = Path(os.environ.get("DORY_LLAMA_BIN", ROOT / "vendor/llama.cpp/build/bin"))
Q = ("Natalia sold clips to 48 of her friends in April, and then she sold half as many "
     "clips in May. How many clips did Natalia sell altogether in April and May?")

pytestmark = pytest.mark.integration


def _cli(gguf, k):
    r = subprocess.run(
        [str(BIN / "llama-completion"), "-m", str(gguf), "-no-cnv", "-n", "48", "--temp", "0",
         "--seed", "0", "-p", Q, "--override-kv", f"dory.num_loops=int:{k}"],
        capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout


def test_loops_change_output(gguf):
    assert _cli(gguf, 1) != _cli(gguf, 4)


def test_embedding_endpoint(gguf):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    p = subprocess.Popen(
        [str(BIN / "llama-server"), "-m", str(gguf), "--port", str(port),
         "--override-kv", "dory.num_loops=int:1", "--embeddings", "--pooling", "last",
         "-ngl", "99"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        url = f"http://127.0.0.1:{port}"
        for _ in range(120):
            try:
                urllib.request.urlopen(url + "/health", timeout=2)
                break
            except Exception:
                assert p.poll() is None, "server died"
                time.sleep(1)
        req = urllib.request.Request(url + "/embedding", json.dumps({"content": "hello"}).encode(),
                                     {"Content-Type": "application/json"})
        out = json.load(urllib.request.urlopen(req, timeout=60))
        vec = out[0]["embedding"] if isinstance(out, list) else out["embedding"]
        while isinstance(vec[0], list):
            vec = vec[0]
        assert len(vec) > 100
    finally:
        p.kill()
        p.wait()
