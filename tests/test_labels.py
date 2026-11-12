import json
from types import SimpleNamespace as NS

from dory.labels import build_labels, class_balance, min_k, non_monotone_rate


def test_min_k():
    assert min_k({1: False, 2: True, 4: True}) == 2
    assert min_k({1: True}) == 1
    assert min_k({1: False, 2: False, 4: False}) is None
    assert min_k({1: False, 2: False}) is None
    assert min_k({1: False, 2: True, 4: False}) == 2


def test_class_balance():
    got = class_balance([{"label_k": k} for k in (1, 1, 2, None)])
    assert got == {1: 2 / 3, 2: 1 / 3}


def test_non_monotone_rate():
    mk = lambda c: {"correct_by_k": c}
    assert non_monotone_rate([mk({1: True, 2: False, 4: True}), mk({1: False, 2: True, 4: True}),
                              mk({1: True, 2: False})]) == 0.5  # incomplete item ignored


def test_build_labels(tmp_path):
    def row(i, k, ok, v="q5", err=None):
        return {"id": i, "source": "gsm8k", "k": k, "variant": v, "output": "", "correct": ok,
                "n_prompt": 10, "n_gen": k, "error": err}
    rows = [row("a", 1, False), row("a", 2, True), row("a", 2, False, err="HTTP 500"),  # retried dup
            row("a", 1, True, v="q8"), row("b", 1, False, err="timeout"), row("b", 2, False), row("b", 4, False)]
    p = tmp_path / "s.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    items = [NS(id=i, source="gsm8k", prompt=f"p{i}") for i in "abc"]
    labels = build_labels(p, items)
    assert len(labels) == 2  # item c has no rows: skipped
    a, b = labels
    assert a["label_k"] == 2 and a["correct_by_k"] == {1: False, 2: True} and a["prompt"] == "pa"
    assert a["tokens_by_k"] == {1: [10, 1], 2: [10, 2]}
    assert b["label_k"] is None
