import random
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from functools import lru_cache

import datasets

MAX_TOKENS = {"gsm8k": 512, "mmlu_pro": 512, "mbpp": 768}
HF = {
    "gsm8k": ("openai/gsm8k", "main"),
    "mmlu_pro": ("TIGER-Lab/MMLU-Pro", None),
    "mbpp": ("google-research-datasets/mbpp", "full"),
}
# (source, hf split, n) in order; each split excludes ids of the earlier ones
SPLITS = {
    "holdout": [("gsm8k", "test", 100), ("mmlu_pro", "test", 100), ("mbpp", "test", 100)],
    "final": [("gsm8k", "test", 35), ("mmlu_pro", "test", 35), ("mbpp", "test", 30)],
    "train": [("gsm8k", "train", 600), ("mmlu_pro", "test", 600), ("mbpp", "train", 300)],
}


@dataclass
class Item:
    id: str
    source: str
    prompt: str
    gold: str
    tests: list[str] | None = None


@lru_cache(maxsize=None)
def _ds(source, split):
    name, cfg = HF[source]
    return datasets.load_dataset(name, cfg, split=split)


def _item(source, split, i, row):
    id_ = f"{source}:{split}:{i}"
    if source == "gsm8k":
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        return Item(id_, source, row["question"] + "\n\nSolve step by step. End with 'The answer is N'.", gold)
    if source == "mmlu_pro":
        opts = "\n".join(f"{chr(65 + j)}. {o}" for j, o in enumerate(row["options"]))
        return Item(id_, source, f"{row['question']}\n{opts}\n\nEnd with 'The answer is (X)'.", row["answer"])
    tests = list(row["test_list"])
    if row["test_setup_code"]:
        tests.insert(0, row["test_setup_code"])
    prompt = f"{row['text']}\n{row['test_list'][0]}\n\nReturn only the function in a ```python code block."
    return Item(id_, source, prompt, "", tests)


def load(source: str, split: str, n: int, seed: int = 0, exclude: set[str] = set()) -> list[Item]:
    ds = _ds(source, split)
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    idx = [i for i in idx if f"{source}:{split}:{i}" not in exclude][:n]
    return [_item(source, split, i, ds[i]) for i in idx]


def load_split(name: str) -> list[Item]:
    out, seen = [], set()
    # earlier splits' ids are excluded; dicts keep holdout, final, train order
    for sp, parts in SPLITS.items():
        items = [it for s, hf, n in parts for it in load(s, hf, n, exclude=seen)]
        seen |= {it.id for it in items}
        if sp == name:
            return items
    raise KeyError(name)


def _f(s):
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def _num(s, fallback=True):
    m = re.findall(r"answer is\s*\$?\s*(-?[\d,]*\.?\d+)", s, re.I)
    m = m or (fallback and re.findall(r"(?<!\d)-?\d[\d,]*(?:\.\d+)?", s))
    return _f(m[-1]) if m else None


def _mbpp(item, output):
    blocks = re.findall(r"```(?:python|py)[ \t]*\r?\n(.*?)```", output, re.I | re.S)
    blocks = blocks or re.findall(r"```[ \t]*\r?\n(.*?)```", output, re.S)
    if not blocks or not item.tests:
        return False
    prog = blocks[-1] + "\n" + "\n".join(item.tests) + "\n"
    # ponytail: subprocess+timeout is not a sandbox; use a container if prompts become untrusted
    with tempfile.TemporaryDirectory() as d:
        try:
            r = subprocess.run([sys.executable, "-I", "-c", prog], cwd=d, timeout=10,
                               capture_output=True)
        except subprocess.TimeoutExpired:  # run() kills the child on timeout
            return False
    return r.returncode == 0


def grade(item: Item, output: str, truncated: bool = False) -> bool:
    if item.source == "gsm8k":
        got, gold = _num(output, not truncated), _num(item.gold)
        return got is not None and gold is not None and got == gold
    if item.source == "mmlu_pro":
        m = re.findall(r"(?i:answer(?:\s+is\s*:?|\s*:))\s*[:\s]*\**\s*\(?\**([A-J])\**\)?(?![A-Za-z])", output)
        if not m and not truncated:
            lines = output.strip().splitlines()
            m = [] if not lines or re.match(r"\s*[-*]\s", lines[-1]) else re.findall(r"^\W*\(?\**([A-J])\**\)?\W*$", lines[-1])
        return bool(m) and m[-1] == item.gold
    return _mbpp(item, output)
