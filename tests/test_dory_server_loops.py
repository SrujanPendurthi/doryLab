"""Patched llama-server: per-request `n_loops` and same-weights self-draft (patches/llama-dory-runtime-loops.patch).

Skips unless DORY_LLAMA_BIN points at a build whose /props has `n_loops_max`. References come from the stock
binary (vendor/llama.cpp, `--override-kv dory.num_loops=int:k`, -np 1) and are cached in results/pi/ref_k{k}.jsonl.
"""
import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from dory.client import CHAT_TEMPLATE_KWARGS
from dory.server import Server, free_port, server_args

pytestmark = pytest.mark.skipif(not os.environ.get("DORY_LLAMA_BIN"), reason="DORY_LLAMA_BIN unset")

ROOT = Path(__file__).resolve().parent.parent
GGUF = ROOT / "models/gguf/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf"
STOCK_BIN = ROOT / "vendor/llama.cpp/build/bin"
REF = ROOT / "results/pi"
CTX = 16384
EXTRA = ["--cache-ram", "2048"]  # same on the stock reference and the patched server
BAD_MSG = "n_loops must be one of 1, 2, 4 and <= n_loops_max"


def _req(url, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url + path, data, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def chat(url, messages, max_tokens, n_loops=None):
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0, "seed": 0,
            "chat_template_kwargs": CHAT_TEMPLATE_KWARGS}  # same body as dory.client.complete
    if n_loops is not None:
        body["n_loops"] = n_loops
    code, d = _req(url, "/v1/chat/completions", body)
    assert code == 200, d
    return d


def text(d):
    return d["choices"][0]["message"]["content"] or ""


@pytest.fixture(scope="module")
def items():
    from dory.evalsets import load_split
    return load_split("wsp_screen")[:20]


def generate_references():
    """Run explicitly before pytest; never run stock and patched servers together."""
    import hashlib
    from dory.evalsets import load_split, MAX_TOKENS
    binary_sha = hashlib.sha256((STOCK_BIN / "llama-server").read_bytes()).hexdigest()
    for k in (1, 2, 4):
        assert not subprocess.run(["pgrep", "-fl", "llama-server"], capture_output=True).stdout
        port = free_port()
        args = server_args(GGUF, k, port, slots=1, ctx_per_slot=CTX, extra=["--jinja", *EXTRA])
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("DORY_LLAMA_BIN", str(STOCK_BIN))
            with Server(args, port) as server:
                rows = []
                for it in load_split("wsp_screen")[:20]:
                    output = text(chat(server.url, it.prompt, MAX_TOKENS[it.source]))
                    rows.append({"id": it.id, "k": k, "output": output,
                                 "max_tokens": MAX_TOKENS[it.source], "prompt": it.prompt,
                                 "binary_sha256": binary_sha, "args": args,
                                 "chat_template_kwargs": CHAT_TEMPLATE_KWARGS})
                    print(f"reference k={k}: {len(rows)}/20 {it.id}", flush=True)
        REF.mkdir(parents=True, exist_ok=True)
        (REF / f"ref_k{k}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture(scope="module")
def refs(items):
    from dory.evalsets import MAX_TOKENS
    out = {}
    for k in (1, 2, 4):
        rows = [json.loads(line) for line in (REF / f"ref_k{k}.jsonl").read_text().splitlines()]
        assert [r["id"] for r in rows] == [it.id for it in items]
        for row, it in zip(rows, items):
            assert row["k"] == k and row["prompt"] == it.prompt
            assert row["max_tokens"] == MAX_TOKENS[it.source]
            assert row["chat_template_kwargs"] == CHAT_TEMPLATE_KWARGS
        out[k] = {r["id"]: r["output"] for r in rows}
    return out


class Patched:
    """At most one patched server alive; restart() stops the old one first."""
    def __init__(self):
        self.s = None

    def restart(self, spec=False, extra=()):
        self.stop()
        port = free_port()
        assert not subprocess.run(["pgrep", "-fl", "llama-server"], capture_output=True).stdout, "another llama-server is running"
        self.s = Server(server_args(GGUF, 4, port, slots=1, ctx_per_slot=CTX, extra=[*EXTRA, *extra], runtime_loops=True,
                                    spec_draft_loops=1 if spec else None), port)
        self.spec = spec
        return self.s.url

    def url(self, spec=False):
        return self.s.url if self.s and self.spec == spec else self.restart(spec)

    def stop(self):
        if self.s:
            self.s.stop()
            self.s = None


@pytest.fixture(scope="module")
def srv():  # committed references are generated before this module starts
    p = Patched()
    try:
        _, props = _req(p.restart(), "/props")
        if "n_loops_max" not in props:
            pytest.skip("DORY_LLAMA_BIN server has no n_loops_max in /props (unpatched build)")
        yield p
    finally:
        p.stop()


@pytest.mark.parametrize("k", [1, 2, 4])
def test_exact_vs_per_k(srv, items, refs, k):
    from dory.evalsets import MAX_TOKENS
    url = srv.url()
    bad = []
    for index, it in enumerate(items, 1):
        got = text(chat(url, it.prompt, MAX_TOKENS[it.source], n_loops=k))
        if got != refs[k][it.id]:
            bad.append(it.id)
        print(f"exact k={k}: {index}/20 {it.id}: {'MATCH' if got == refs[k][it.id] else 'DIFFER'}", flush=True)
    assert not bad, f"k={k}: {len(bad)}/20 differ from the stock per-k reference: {bad}"


def test_timings_and_props(srv):
    url = srv.url()
    assert _req(url, "/props")[1]["n_loops_max"] == 4
    for k in (1, 2, 4):
        assert chat(url, "Say hi.", 4, n_loops=k)["timings"]["n_loops"] == k
    assert chat(url, "Say hi.", 4)["timings"]["n_loops"] == 4  # no n_loops: GGUF num_loops


def test_bad_n_loops(srv):
    url = srv.url()
    bodies = {
        "/completion": {"prompt": "hi", "n_predict": 2},
        "/v1/completions": {"prompt": "hi", "max_tokens": 2},
        "/v1/chat/completions": {"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 4},
    }
    for path, base in bodies.items():
        for v in (0, 3, 5, "2", None, True, 2.0, -1, 2**64 - 1):
            code, d = _req(url, path, {**base, "n_loops": v})
            assert code == 400 and d["error"]["message"] == BAD_MSG, (path, v, code, d)
        code, d = _req(url, path, {**base, "n_loops": 2})
        assert code == 200 and d["timings"]["n_loops"] == 2, (path, code, d)


def _turns(items):
    p = items[0].prompt
    return p, lambda a1: [{"role": "user", "content": p}, {"role": "assistant", "content": a1},
                          {"role": "user", "content": "Now check your answer and restate it in one sentence."}]


def test_switch_matches_fresh(srv, items):
    p, second = _turns(items)
    url = srv.restart()
    a1 = text(chat(url, p, 128, n_loops=1))
    switched = chat(url, second(a1), 128, n_loops=4)
    assert switched["timings"]["cache_n"] == 0
    got = text(switched)  # slot holds P's k=1 KV: must reprocess at k=4
    url = srv.restart()
    want = text(chat(url, second(a1), 128, n_loops=4))
    assert got == want


def test_prompt_cache_keyed_by_loops(srv, items):
    p, q = items[1].prompt, items[2].prompt
    url = srv.restart()
    chat(url, p, 64, n_loops=2)
    chat(url, q, 64, n_loops=2)  # P leaves the slot for the prompt cache (--cache-ram 2048)
    switched = chat(url, p, 128, n_loops=4)
    assert switched["timings"]["cache_n"] == 0
    got = text(switched)
    url = srv.restart()
    want = text(chat(url, p, 128, n_loops=4))
    assert got == want


def test_spec_identical(srv, items, refs):
    from dory.evalsets import MAX_TOKENS
    url = srv.url(spec=True)
    low = chat(url, "Say hi.", 8, n_loops=1)
    assert low["timings"]["n_loops"] == 1 and low["timings"].get("draft_n", 0) == 0
    bad, drafted = [], []
    for it, k in [(it, 4) for it in items[:10]] + [(it, 2) for it in items[10:20]]:
        d = chat(url, it.prompt, MAX_TOKENS[it.source], n_loops=k)
        if text(d) != refs[k][it.id]:
            bad.append((k, it.id))
        if k == 4:
            drafted.append(d["timings"].get("draft_n", 0))
        print(f"spec k={k} {it.id}: {'MATCH' if text(d) == refs[k][it.id] else 'DIFFER'}, draft_n={d['timings'].get('draft_n', 0)}", flush=True)
    assert not bad, f"speculative outputs differ: {bad}"
    assert all(n > 0 for n in drafted), drafted


def test_memory(srv, items):
    url = srv.restart(spec=True, extra=["--verbosity", "4"])
    chat(url, items[0].prompt, 64, n_loops=4)
    rss_gb = srv.s.rss_mb() * 1024**2 / 1_000_000_000
    srv.s._err.seek(0)
    log = srv.s._err.read().decode(errors="replace")
    lines = [l for l in log.splitlines() if re.search(r"llama_kv_cache|compute buffer", l)]
    print(f"\nRSS after one k=4 request: {rss_gb:.2f} GB\n" + "\n".join(lines))
    assert any("KV buffer size" in line for line in lines), "KV sizes missing from server log"
    assert any("compute buffer size" in line for line in lines), "compute sizes missing from server log"
    assert rss_gb <= 9, f"RSS {rss_gb:.2f} GB > 9 GB"
