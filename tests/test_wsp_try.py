import json
from types import SimpleNamespace

import pytest

from scripts import wsp_try as W

OPS = [{"tensor": "a", "op": "set", "index": [[0, 0]], "value": [1.0]}]


def edit(i="e1", ops=OPS):
    return {"id": i, "hypothesis": "h", "evidence": ["x"], "ops": ops}


@pytest.fixture
def env(tmp_path, monkeypatch):
    (tmp_path / "models/bf16").mkdir(parents=True)
    st = {"n_changed": 1, "ndiff": 1, "screen": True, "acc": (True, "accept: ok")}

    def fake_apply(e, base, out):
        out.mkdir(parents=True)
        return SimpleNamespace(n_requested=1, n_changed=st["n_changed"], tensors={"a": 1})
    monkeypatch.setattr(W, "apply_edit", fake_apply)
    monkeypatch.setattr(W, "random_control", lambda e, b, seed, npz_dir: (npz_dir.mkdir(parents=True), {"ops": []})[1])
    monkeypatch.setattr(W, "to_gguf", lambda d, out: out.write_text("g"))
    monkeypatch.setattr(W, "gguf_diff_count", lambda a, b: st["ndiff"])
    monkeypatch.setattr(W, "evaluate", lambda g, s: [])
    monkeypatch.setattr(W, "summarize", lambda r: {"acc": 0})
    monkeypatch.setattr(W, "compare", lambda a, b: {"dacc_pts": 0.0})
    monkeypatch.setattr(W, "screen_pass", lambda a, b: st["screen"])
    monkeypatch.setattr(W, "accept", lambda *a: st["acc"])
    return tmp_path, st


def ledger(root):
    return (root / "research/wsp/ledger.md").read_text()


def test_noop_rejected_without_flag(env):
    root, st = env
    st["n_changed"] = 0
    assert W.run_trial(edit(), root) == (False, "no-op edit")
    assert W.run_trial(edit("e2"), root, allow_noop=True)[1] != "no-op edit"


def test_quant_loss_rejects(env):
    root, st = env
    st["ndiff"] = 0
    assert W.run_trial(edit(), root) == (False, "edit lost in quantization")


def test_screen_fail_cleans_up(env):
    root, st = env
    st["screen"] = False
    assert not W.run_trial(edit(), root)[0]
    m = root / "results/wsp/models"
    assert not (m / "e1").exists() and not (m / "e1.gguf").exists()
    assert "## e1" in ledger(root)


def test_accept_updates_best(env):
    root, _ = env
    assert W.run_trial(edit(), root)[0]
    b = json.loads((root / "research/wsp/best.json").read_text())
    assert b == {"model_dir": "results/wsp/models/e1", "gguf": "results/wsp/models/e1.gguf", "accepted": ["e1"]}
    m = root / "results/wsp/models"
    assert sorted(p.name for p in m.iterdir()) == ["e1", "e1.gguf"]


def test_duplicate_id_refused(env):
    root, _ = env
    W.run_trial(edit(), root)
    with pytest.raises(SystemExit):
        W.run_trial(edit(), root)


def test_hours_cumulative(env):
    root, _ = env
    W.run_trial(edit(), root)
    p = root / "research/wsp/ledger.md"
    p.write_text(p.read_text().replace("<!-- hours: 0", "<!-- hours: 2.0 -->\n<!-- x: 0", 1))
    W.run_trial(edit("e2", OPS), root)
    assert "cumulative: 2.0" in ledger(root).split("## e2")[1]
