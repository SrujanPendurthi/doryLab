import json
from types import SimpleNamespace

import pytest

from scripts import wsp_try as W

OPS = [{"tensor": "model.layers.0.self_attn.q_proj.weight", "op": "set", "index": [[0, 0]], "value": [1.0]}]


def edit(i="e1", ops=OPS):
    return {"id": i, "hypothesis": "h", "evidence": ["x"], "ops": ops}


@pytest.fixture
def env(tmp_path, monkeypatch):
    (tmp_path / "models/bf16").mkdir(parents=True)
    st = {"n_changed": 1, "ndiff": 1, "screen": True, "acc": (True, "accept: ok")}

    def fake_apply(e, base, out):
        out.mkdir(parents=True)
        return SimpleNamespace(n_requested=1, n_changed=st["n_changed"], tensors=["a"], delta_l2={"a": 1.0 if st["n_changed"] else 0.0})
    monkeypatch.setattr(W, "apply_edit", fake_apply)
    monkeypatch.setattr(W, "random_control", lambda e, b, seed, npz_dir: (npz_dir.mkdir(parents=True), {"ops": []})[1])
    monkeypatch.setattr(W, "to_gguf", lambda d, out: out.write_text("g"))
    monkeypatch.setattr(W, "gguf_diff_count", lambda a, b: {"blk.0.attn_q.weight"} if st["ndiff"] else set())
    monkeypatch.setattr(W, "gguf_name", lambda n: "blk.0.attn_q.weight")
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
    p = root / "research/wsp/ledger.md"
    p.parent.mkdir(parents=True)
    p.write_text("\n## old\n<!-- hours: 2.000000 -->\n")
    W.run_trial(edit("e2"), root)
    assert "cumulative: 2.0" in ledger(root).split("## e2")[1]


def test_bad_id_refused(env):
    with pytest.raises(SystemExit):
        W.run_trial(edit("../x"), env[0])


def test_noop_skips_gguf(env, monkeypatch):
    root, st = env
    st["n_changed"] = 0
    monkeypatch.setattr(W, "to_gguf", lambda d, o: pytest.fail("converted"))
    assert W.run_trial(edit(), root)[1] == "no-op edit"


def test_error_cleans_and_logs(env, monkeypatch):
    root, _ = env
    def boom(g, s):
        raise RuntimeError("server died")
    monkeypatch.setattr(W, "evaluate", boom)
    with pytest.raises(RuntimeError):
        W.run_trial(edit(), root)
    m = root / "results/wsp/models"
    assert list(m.iterdir()) == []
    assert "ERROR" in ledger(root) and "server died" in ledger(root)


def test_accept_keeps_previous_best(env):
    root, _ = env
    assert W.run_trial(edit("e1"), root)[0]
    assert W.run_trial(edit("e2"), root)[0]
    m = root / "results/wsp/models"
    assert (m / "e1").exists() and (m / "e1.gguf").exists() and (m / "e2").exists()


def test_real_gguf_diff_count(tmp_path):
    import numpy as np
    g = W._gguf()
    def mk(p, w):
        wr = g.GGUFWriter(str(p), "dory")
        wr.add_tensor("x", np.ones((4, 4), dtype=np.float32))
        wr.add_tensor("y", w)
        wr.write_header_to_file(); wr.write_kv_data_to_file(); wr.write_tensors_to_file(); wr.close()
    mk(tmp_path / "a.gguf", np.zeros((4, 4), dtype=np.float32))
    mk(tmp_path / "b.gguf", np.full((4, 4), 2, dtype=np.float32))
    assert W.gguf_diff_count(tmp_path / "a.gguf", tmp_path / "b.gguf") == {"y"}
