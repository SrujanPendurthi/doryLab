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


@pytest.mark.parametrize("fence", ["```py\n", "```Python\n", "```python\r\n", "```python \n"])
def test_mbpp_fence_variants(fence):
    assert grade(mbpp(["assert f()==1"]), f"{fence}def f(): return 1\n```")


def test_mmlu_prose_letters_not_answers():
    assert not grade(Item("x", "mmlu_pro", "q", "I"), "I don't know")
    assert not grade(Item("x", "mmlu_pro", "q", "D"), "The answer is Dogs")
    assert grade(Item("x", "mmlu_pro", "q", "B"), "reasoning\n**B**")


@pytest.mark.parametrize("out,gold", [
    ("The answer is 18. (100% sure)", "18"),
    ("-7", "-7"),
    ("5-3=2. The answer is 2", "2"),
    ("The answer is $1,234.50", "1234.5"),
])
def test_gsm8k_edge(out, gold):
    assert grade(Item("x", "gsm8k", "q", gold), out)


def test_load_deterministic(monkeypatch):
    import dory.evalsets as ev
    rows = [{"question": f"q{i}", "answer": f"#### {i}"} for i in range(50)]
    monkeypatch.setattr(ev, "_ds", lambda s, sp: rows)
    a, b = ev.load("gsm8k", "test", 10), ev.load("gsm8k", "test", 10)
    assert [i.id for i in a] == [i.id for i in b]
