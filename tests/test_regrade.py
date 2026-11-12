import json

import dory.evalsets as E
from scripts import regrade


def test_regrade(tmp_path, monkeypatch):
    items = [E.Item("a", "gsm8k", "q", "42"), E.Item("b", "gsm8k", "q", "42"), E.Item("c", "gsm8k", "q", "1")]
    monkeypatch.setattr(regrade, "load_split", lambda s: items)
    rows = [{"id": "a", "source": "gsm8k", "output": "= 42", "n_gen": 512, "correct": True, "error": None},
            {"id": "b", "source": "gsm8k", "output": "= 42", "n_gen": 10, "correct": False, "error": None},
            {"id": "c", "source": "gsm8k", "output": "x", "n_gen": 5, "correct": True, "error": "boom"}]
    p = tmp_path / "sweep_holdout.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    regrade.main([str(p)])
    out = [json.loads(l) for l in p.read_text().splitlines()]
    assert [r["correct"] for r in out] == [False, True, True]
