from contextlib import nullcontext
from types import SimpleNamespace

import dory.sweep as sw
from dory.client import ServerError
from dory.evalsets import Item


def mk(i):
    return Item(f"i{i}", "gsm8k", f"p{i}", "1")


def row(i, k, correct=True, variant="q5"):
    return {"id": f"i{i}", "k": k, "variant": variant, "correct": correct}


def test_pending_full():
    items = [mk(0), mk(1)]
    done = [row(0, 1), row(1, 1, variant="other")]
    assert [i.id for i in sw.pending(items, 1, "q5", done, False)] == ["i1"]


def test_pending_lazy():
    items = [mk(0), mk(1)]
    done = [row(0, 1, True), row(1, 1, False)]
    assert [i.id for i in sw.pending(items, 2, "q5", done, True)] == ["i1"]
    assert [i.id for i in sw.pending(items, 2, "q5", done, False)] == ["i0", "i1"]


def test_read_rows_skips_torn_line(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text('{"id": "a"}\n{"id": "b"}\n{"id": "x", "k"')
    assert len(sw.read_rows(p)) == 2


def patch_server(monkeypatch, started):
    def srv(args, port, **kw):
        started.append(args)
        return nullcontext(SimpleNamespace(url=args))  # url carries k via args
    monkeypatch.setattr(sw, "Server", srv)
    monkeypatch.setattr(sw, "server_args", lambda gguf, k, port, **kw: k)
    monkeypatch.setattr(sw, "free_port", lambda: 1)


def test_error_row_recorded(monkeypatch, tmp_path):
    def boom(*a, **kw):
        raise ServerError("ctx")
    monkeypatch.setattr(sw, "complete", boom)
    patch_server(monkeypatch, [])
    out = tmp_path / "o.jsonl"
    sw.run_sweep([mk(0)], [1], "q5", tmp_path / "g", out, 1)
    (r,) = sw.read_rows(out)
    assert r["error"] == "ctx" and r["correct"] is False


def test_grade_exception_recorded(monkeypatch, tmp_path):
    monkeypatch.setattr(sw, "complete", lambda *a, **kw: SimpleNamespace(text="t", n_prompt=1, n_gen=1))
    monkeypatch.setattr(sw, "grade", lambda it, o, t=False: 1 / 0)
    patch_server(monkeypatch, [])
    out = tmp_path / "o.jsonl"
    sw.run_sweep([mk(0)], [1], "q5", tmp_path / "g", out, 1)
    (r,) = sw.read_rows(out)
    assert r["error"].startswith("grade:") and r["correct"] is False


def test_lazy_run_sweep(monkeypatch, tmp_path):
    started, calls = [], []

    def fake(url, prompt, max_tokens, timeout=900):
        calls.append((url, prompt))
        return SimpleNamespace(text=f"k{url}", n_prompt=1, n_gen=1)  # url == k

    monkeypatch.setattr(sw, "complete", fake)
    # i0 correct already at k=1; i1 only correct at k=2
    monkeypatch.setattr(sw, "grade", lambda it, o, t=False: it.id == "i0" or o == "k2")
    patch_server(monkeypatch, started)
    out = tmp_path / "o.jsonl"
    sw.run_sweep([mk(0), mk(1)], [1, 2], "q5", tmp_path / "g", out, 2, lazy=True)
    assert sorted(calls) == [(1, "p0"), (1, "p1"), (2, "p1")]
    rows = sw.read_rows(out)
    assert {(r["id"], r["k"], r["correct"]) for r in rows} == {("i0", 1, True), ("i1", 1, False), ("i1", 2, True)}
    # rerun: nothing pending, no server started
    started.clear()
    sw.run_sweep([mk(0), mk(1)], [1, 2], "q5", tmp_path / "g", out, 2, lazy=True)
    assert started == []


def test_resume_after_torn_line(monkeypatch, tmp_path):
    out = tmp_path / "o.jsonl"
    out.write_text('{"id": "x", "k"')
    monkeypatch.setattr(sw, "complete", lambda *a, **kw: SimpleNamespace(text="t", n_prompt=1, n_gen=1))
    monkeypatch.setattr(sw, "grade", lambda it, o, t=False: True)
    patch_server(monkeypatch, [])
    sw.run_sweep([mk(0)], [1], "q5", tmp_path / "g", out, 1)
    assert [r["id"] for r in sw.read_rows(out)] == ["i0"]


def test_read_rows_skips_non_dict(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text('[1]\n3\n{"id": "a"}\n')
    assert len(sw.read_rows(p)) == 1


def err(i, k, e, variant="q5"):
    return {"id": f"i{i}", "k": k, "variant": variant, "correct": False, "error": e}


def test_pending_error_policy():
    items = [mk(0), mk(1), mk(2)]
    done = [err(0, 1, "timed out"), err(1, 1, "HTTP 400 ctx"), err(2, 1, "grade: boom")]
    assert [i.id for i in sw.pending(items, 1, "q5", done, False)] == ["i0"]


def test_latest_rows_prefers_clean():
    ok = {"id": "i0", "k": 1, "variant": "q5", "correct": True, "error": None}
    e = err(0, 1, "timed out")
    key = ("i0", 1, "q5")
    assert sw.latest_rows([e, ok])[key] is ok
    assert sw.latest_rows([ok, e])[key] is ok
    assert sw.latest_rows([e])[key] is e
