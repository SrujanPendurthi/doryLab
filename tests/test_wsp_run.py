import json
import pytest
from dory import evalsets
from dory.wsp import run


def rows(spec, src="gsm8k"):
    """spec: list of (correct, n_gen); ids 0..n-1."""
    return [{"id": f"{src}{i}", "source": src, "k": 1, "variant": "wsp", "output": "", "correct": c,
             "n_prompt": 1, "n_gen": g, "error": None} for i, (c, g) in enumerate(spec)]


def test_summarize_truncation_and_per_source():
    r = rows([(True, 10), (False, 512)]) + rows([(True, 768)], "mbpp")
    s = run.summarize(r)
    assert s["n"] == 3 and s["acc"] == pytest.approx(2 / 3)
    assert s["trunc_rate"] == pytest.approx(2 / 3)
    assert s["per_source"]["gsm8k"] == {"acc": 0.5, "n": 2, "mean_tokens": 261}
    assert s["per_source"]["mbpp"]["acc"] == 1
    r[0]["error"] = "boom"
    with pytest.raises(ValueError):
        run.summarize(r)


def test_compare_pairs_by_id():
    b = rows([(False, 10), (True, 10), (False, 10), (False, 10)])
    e = rows([(True, 5), (True, 5), (False, 5), (False, 5)])
    c = run.compare(b, list(reversed(e)))
    assert c["dacc_pts"] == pytest.approx(25) and c["tokens_ratio"] == 0.5
    assert c["per_source_dacc_pts"] == {"gsm8k": pytest.approx(25)}
    with pytest.raises(ValueError):
        run.compare(b, e[:3])


def test_screen_pass_rules():
    base = rows([(True, 100)] * 10 + [(False, 100)] * 10)
    assert run.screen_pass(base, rows([(True, 105)] * 10 + [(False, 105)] * 10))
    assert not run.screen_pass(base, rows([(True, 100)] * 9 + [(False, 100)] * 11))
    assert not run.screen_pass(base, rows([(True, 106)] * 10 + [(False, 106)] * 10))


def pct(n_correct, n=100, g=100, src="gsm8k"):
    return rows([(True, g)] * n_correct + [(False, g)] * (n - n_correct), src)


def test_accept_paths():
    base, arc = pct(50), pct(50, src="arc")
    # accuracy path: +10 pts on 100 paired items, control flat
    ok, why = run.accept(base, pct(60), pct(50), arc, arc)
    assert ok and "accuracy" in why
    # token path: same acc, tokens 0.8x, control full tokens
    ok, why = run.accept(base, pct(50, g=80), pct(50), arc, arc)
    assert ok and "token" in why
    # control within 2 pts
    assert not run.accept(base, pct(60), pct(59), arc, arc)[0]
    # one source drops 3.1 pts while overall gains
    two = pct(50, 100) + pct(50, 100, src="mmlu_pro")
    ed = pct(70, 100) + pct(50, 100, src="mmlu_pro")
    base2 = two
    ed[100:] = rows([(True, 100)] * 47 + [(False, 100)] * 53, "mmlu_pro")  # -3 exactly: ok
    assert run.accept(base2, ed, two, arc, arc)[0]
    ed[100:] = rows([(True, 100)] * 46 + [(False, 100)] * 54, "mmlu_pro")  # -4: reject
    assert "mmlu_pro" in run.accept(base2, ed, two, arc, arc)[1]
    # ARC drops 3.1+ pts
    ok, why = run.accept(base, pct(60), pct(50), arc, pct(46, src="arc"))
    assert not ok and "ARC" in why
    ok, _ = run.accept(base, pct(60), pct(50), arc, pct(47, src="arc"))
    assert ok


class Item:
    def __init__(self, i):
        self.id, self.source = i, "gsm8k"


def test_cache_meta_mismatch_raises(tmp_path, monkeypatch):
    g = tmp_path / "m.gguf"
    g.write_bytes(b"abc")
    monkeypatch.setattr(run, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(evalsets, "load_split", lambda s: [Item("a")])
    calls = []

    def fake(items, ks, variant, gguf, out, slots):
        calls.append(1)
        with open(out, "a") as f:
            f.write(json.dumps(rows([(True, 5)])[0] | {"id": "a"}) + "\n")

    monkeypatch.setattr(run, "run_sweep", fake)
    out = run.evaluate(g, "wsp_screen")
    assert len(out) == 1 and out[0]["id"] == "a"
    meta = json.loads(next((tmp_path / "cache").glob("*/meta.json")).read_text())
    assert meta["gguf"] == str(g) and len(meta["sha256"]) == 64
    monkeypatch.setattr(evalsets, "MAX_TOKENS", {**evalsets.MAX_TOKENS, "gsm8k": 1})
    with pytest.raises(RuntimeError):
        run.evaluate(g, "wsp_screen")
