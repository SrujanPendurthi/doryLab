import time

import pytest

from dory.evalsets import Item, grade, load_split

SOURCES = ["gsm8k", "mmlu_pro", "mbpp"]


def test_gsm8k_grade():
    it = Item("gsm8k:test:0", "gsm8k", "q", "1234")
    assert grade(it, "... The answer is 1,234.")
    assert not grade(it, "The answer is 12")


def test_mmlu_grade():
    it = Item("mmlu_pro:test:0", "mmlu_pro", "q", "C")
    assert grade(it, "The answer is (C)")
    assert not grade(Item("x", "mmlu_pro", "q", "D"), "The answer is (C)")


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("out", ["", "I don't know", "```python\ndef f(:\n"])
def test_grade_garbage_is_false(source, out):
    it = Item("x", source, "q", "5", ["assert f()==5"] if source == "mbpp" else None)
    assert grade(it, out) is False


def mbpp(tests):
    return Item("mbpp:test:0", "mbpp", "q", "", tests)


def test_mbpp_pass_and_fail():
    code = "```python\ndef add(a,b): return a+b\n```"
    assert grade(mbpp(["assert add(1,2)==3"]), code)
    assert not grade(mbpp(["assert add(1,2)==4"]), code)


def test_mbpp_timeout_false():
    t = time.time()
    assert not grade(mbpp(["assert True"]), "```python\nwhile True: pass\n```")
    assert time.time() - t < 15


@pytest.mark.network
def test_splits_disjoint():
    s = {n: {i.id for i in load_split(n)} for n in ["holdout", "final", "train"]}
    assert [len(s[n]) for n in ["holdout", "final", "train"]] == [300, 100, 1500]
    assert not (s["holdout"] & s["final"] or s["holdout"] & s["train"] or s["final"] & s["train"])
