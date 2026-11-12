import json

import pytest

from dory.config import KS
from dory.evaluate import (fixed, modeled_seconds, oracle, paired_bootstrap,
                           policy_metrics, random_matched, report)

SPEED = {1: {"prompt_tps": 100.0, "gen_tps": 10.0},
         2: {"prompt_tps": 50.0, "gen_tps": 10.0},
         4: {"prompt_tps": 25.0, "gen_tps": 10.0}}
TOK = {k: [100, 20] for k in KS}
LABELS = [
    {"id": "a", "source": "gsm8k", "prompt": "p", "label_k": 1,
     "correct_by_k": {1: True, 2: True, 4: True}, "tokens_by_k": TOK},
    {"id": "b", "source": "gsm8k", "prompt": "p", "label_k": 2,
     "correct_by_k": {1: False, 2: True, 4: True}, "tokens_by_k": TOK},
    {"id": "c", "source": "mbpp", "prompt": "p", "label_k": None,
     "correct_by_k": {1: False, 2: False, 4: False}, "tokens_by_k": TOK},
]


def test_modeled_seconds():
    assert modeled_seconds(1000, 300, 2, {2: {"prompt_tps": 500, "gen_tps": 30}}) == 12.0


def test_oracle_dominates():
    o = policy_metrics(LABELS, oracle(LABELS), SPEED)
    f = [policy_metrics(LABELS, fixed(LABELS, k), SPEED) for k in KS]
    assert o["acc"] >= max(m["acc"] for m in f)
    assert o["mean_k"] <= f[-1]["mean_k"]
    assert oracle(LABELS) == {"a": 1, "b": 2, "c": 1}


def test_policy_metrics_values():
    m = policy_metrics(LABELS, fixed(LABELS, 2), SPEED, extra_seconds={i["id"]: 1.0 for i in LABELS})
    assert m == pytest.approx({"acc": 2 / 3, "mean_k": 2.0, "mean_seconds": 5.0})


def test_string_keys_after_json_roundtrip():
    rt = json.loads(json.dumps(LABELS))
    assert policy_metrics(rt, fixed(rt, 2), SPEED) == policy_metrics(LABELS, fixed(LABELS, 2), SPEED)


def test_bootstrap_identical():
    x = [True, False, True, True]
    r = paired_bootstrap(x, x)
    assert r == (0.0, 0.0, 0.0) and all(type(v) is float for v in r)


def test_bootstrap_detects_gain():
    m, lo, hi = paired_bootstrap([True] * 50, [False] * 50)
    assert (m, lo, hi) == (1.0, 1.0, 1.0)


def test_random_matched_mean():
    labs = [{"id": str(i)} for i in range(10000)]
    c = random_matched(labs, 2.5, seed=1)
    assert set(c.values()) <= set(KS)
    assert abs(sum(c.values()) / len(c) - 2.5) < 0.05
    assert set(random_matched(labs, 2, seed=1).values()) == {2}


def test_report():
    out = report(LABELS, {"fixed2": (fixed(LABELS, 2), None), "oracle": (oracle(LABELS), None)}, SPEED)
    assert "policy | acc | mean_k | mean_seconds | Δacc vs fixed2 [95% CI]" in out
    assert "oracle" in out and "fixed2" in out
    out2 = report(LABELS, {"oracle": (oracle(LABELS), None)}, SPEED)
    assert "oracle" in out2
