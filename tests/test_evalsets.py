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


@pytest.mark.parametrize("out", ["answer: C", "Answer: C", "answer is: C", "The answer is **C**",
                                 "**Answer: C**", "answer is **(C)**"])
def test_mmlu_formats(out):
    assert grade(Item("x", "mmlu_pro", "q", "C"), out)


def test_mbpp_plain_fence_fallback():
    assert grade(mbpp(["assert f()==1"]), "```\ndef f(): return 1\n```")
    # python block wins over a later plain block
    assert grade(mbpp(["assert f()==1"]), "```python\ndef f(): return 1\n```\n```\nnot code\n```")


@pytest.mark.parametrize("out", ["answer I think so", "the answer B is wrong"])
def test_mmlu_loose_answer_prose_rejected(out):
    assert not grade(Item("x", "mmlu_pro", "q", "I" if out.startswith("answer I") else "B"), out)


def test_truncated_grading():
    g = Item("x", "gsm8k", "q", "42")
    assert not grade(g, "so 6*7 = 42", truncated=True)
    assert grade(g, "so 6*7 = 42")
    assert grade(g, "The answer is 42", truncated=True)
    m = Item("x", "mmlu_pro", "q", "C")
    assert not grade(m, "reasoning\nC", truncated=True)
    assert grade(m, "reasoning\nC")
    assert grade(m, "The answer is C", truncated=True)
    assert not grade(Item("x", "mmlu_pro", "q", "J"), "- J. .")


def test_wsp_screen_subset():
    scr, ho = load_split("wsp_screen"), {it.id for it in load_split("holdout")}
    assert len(scr) == 60 and all(it.id in ho for it in scr)
    assert [sum(it.source == s for it in scr) for s in SOURCES] == [20, 20, 20]


def test_holdout_unchanged():
    ids = [it.id for it in load_split("holdout")]
    assert ids[::50] == ["gsm8k:test:1101", "gsm8k:test:1011", "mmlu_pro:test:6005",
                         "mmlu_pro:test:5573", "mbpp:test:419", "mbpp:test:116"]


def test_wsp_final_sealed(monkeypatch):
    monkeypatch.delenv("WSP_UNSEAL", raising=False)
    with pytest.raises(PermissionError):
        load_split("wsp_final")
    monkeypatch.setenv("WSP_UNSEAL", "1")
    fin = load_split("wsp_final")
    used = {it.id for s in ("holdout", "final", "train") for it in load_split(s)}
    assert len(fin) == 300 and not used & {it.id for it in fin}


def test_arc_numeric_labels_mapped():
    from dory.evalsets import _item
    row = {"question": "q", "choices": {"label": ["1", "2", "3", "4"], "text": list("wxyz")}, "answerKey": "2"}
    it = _item("arc", "test", 0, row)
    assert it.gold == "B" and "\nA. w\nB. x\nC. y\nD. z" in it.prompt


def test_arc_item_and_grade():
    arc = load_split("arc")
    assert len(arc) == 200
    assert all(it.gold in "ABCDE" and "\nA. " in it.prompt and it.prompt.endswith("End with 'The answer is (X)'.") for it in arc)
    assert grade(Item("arc:test:0", "arc", "q", "B"), "The answer is (B)")
