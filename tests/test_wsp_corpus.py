import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "DoryConfig"):
    pytest.skip("transformers build without dory (use .venv-wsp)", allow_module_level=True)

from scripts import wsp_corpus as wc
from dory.wsp.model import DoryTorch

CFG = dict(vocab_size=64, hidden_size=64, num_hidden_layers=6, intermediate_size=96, num_attention_heads=4,
           num_key_value_heads=2, head_dim=16, n_input_layers=2, n_recurrent_layers=2, n_output_layers=2,
           n_recurrent_loops=2, layer_types=["full_attention", "mlp"] * 3, pad_token_id=0)


def test_taxonomy_rules():
    loop = "intro " + "a b c d e f g h " * 3 + "tail"
    c, gram, off = wc.classify(loop, "gsm8k")
    assert c == "loop" and gram.startswith("a b c d") and off == loop.index("a b c d")
    assert wc.classify(" ".join(f"w{i}" for i in range(200)), "gsm8k")[0] == "overlong"
    assert wc.classify("work work. The answer is 5", "gsm8k")[0] == "other"
    assert wc.classify("def f(): pass " * 1 + "```python\ncode", "mbpp")[0] == "overlong"
    assert wc.classify("```python\ncode\n```", "mbpp")[0] == "other"


def test_target_position_gsm8k_mmlu_mbpp():
    g = ["So", " the", " answer", " is", " $", "18", "."]
    assert wc.target_position("gsm8k", g) == 4
    assert wc.target_position("gsm8k", ["no", " phrase"]) is None
    m = ["x", " answer", " is", " (", "B", ")", " "]
    assert wc.target_position("mmlu_pro", m) == 4
    b = ["Here", " ```", "python", "\n", "def", " f"]
    assert wc.target_position("mbpp", b) == 4  # whitespace token skipped
    assert wc.char_to_token(["ab", "cd", "e"], 3) == 1


def test_stage_idempotent(tmp_path):
    (tmp_path / "results/wsp").mkdir(parents=True)
    s = {"acc": 0.5, "n": 4, "mean_tokens": 10.0, "trunc_rate": 0.0, "per_source": {"gsm8k": {"acc": 0.5, "n": 4, "mean_tokens": 10.0}}}
    (tmp_path / "results/wsp/baselines.json").write_text(json.dumps({"screen_k1": s, "noise0": dict(s, n_changed=3)}))
    wc.stage_baselines(tmp_path)
    p = tmp_path / "research/wsp/corpus/00-baselines.md"
    assert "noise0" in p.read_text()
    p.write_text("sentinel")
    wc.stage_baselines(tmp_path)
    assert p.read_text() == "sentinel"


def test_probe_null_near_chance():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    X = rng.normal(size=(300, 20))
    X[:, 0] += 3 * y
    r = wc.cv_auc(X, y)
    assert r["auc_mean"] > 0.9 and 0.35 <= r["null_mean"] <= 0.65
    assert wc.cv_auc(X, np.zeros(300)) is None


def test_ablation_stage_tiny():
    model = DoryTorch.from_config(CFG)
    rng = np.random.default_rng(1)
    mk = lambda: (list(rng.integers(1, 64, 12)), 6)
    groups = {"trunc": [mk(), mk()], "fin": [mk(), mk()]}
    res = wc.run_ablations(model, groups, top_attn=2, top_mlp=2, n_neurons=4)
    assert all(f"layer.{p}" in res for p in range(6))
    assert sum(c.startswith("head.") for c in res) == 2 * 4
    assert sum(c.startswith("neurons.") for c in res) == 2
    assert all(set(v) == {"trunc", "fin", "diff"} for v in res.values())


@pytest.mark.skipif(not os.environ.get("WSP_SERVER_URL"), reason="WSP_SERVER_URL unset")
def test_template_matches_server():
    from dory import evalsets
    from dory.wsp.model import template_ids
    tok = wc.get_tok()
    for it in evalsets.load_split("train")[:5]:
        assert wc.prompt_ids(tok, it.prompt) == template_ids(os.environ["WSP_SERVER_URL"], it.prompt)


class FakeTok:
    def apply_chat_template(self, msgs, add_generation_prompt, tokenize, enable_thinking):
        return "U:" + msgs[0]["content"] + "\nA:"

    def encode(self, text, add_special_tokens=False):
        return list(text.encode())

    def decode(self, ids):
        return bytes(ids).decode()


CFG256 = dict(CFG, vocab_size=256)


def _rows(n):
    from dory.evalsets import Item
    rows, items = [], []
    for i in range(n):
        src = "gsm8k" if i % 2 else "mmlu_pro"
        out = "work. The answer is 5." if src == "gsm8k" else "work. The answer is (B) ok"
        trunc = i % 4 in (0, 1)
        rows.append({"id": f"x{i}", "source": src, "output": out, "n_gen": 512 if trunc else 20,
                     "correct": i % 3 == 0, "error": None})
        items.append(Item(f"x{i}", src, "q" * (i + 1), "A"))
    return rows, items


def test_stage_lens_tiny(tmp_path):
    rows, items = _rows(6)
    model = DoryTorch.from_config(CFG256)
    wc.stage_lens(rows, items, tmp_path, model, FakeTok(), n=6, loop_set=(1, 2))
    z = np.load(tmp_path / "results/wsp/corpus/lens.npz")
    assert len(z.files) == 2 * 2 * 1  # only finished rows (i%4 in 2,3): x2 mmlu_pro, x3 gsm8k; two loops each
    assert z["x3|L1"].shape == (6,) and z["x3|L2"].shape == (8,)
    assert "02-lens" in str(next((tmp_path / "research/wsp/corpus").glob("02-*")))


def test_stage_probes_tiny(tmp_path):
    rows, items = _rows(24)
    model = DoryTorch.from_config(CFG256)
    wc.stage_probes(rows, items, tmp_path, model, FakeTok(), tax=[])
    res = json.loads((tmp_path / "results/wsp/corpus/probes.json").read_text())
    for name in ("will_truncate", "will_be_correct"):
        d = res["prompt_end"][name]
        assert set(d) == {"source_onehot", "all", "within"} and set(d["within"]) == {"gsm8k", "mmlu_pro"}
        assert d["source_onehot"]["auc_mean"] is not None
    assert (tmp_path / "research/wsp/corpus/03-probes.md").exists()


def test_ablate_clean_arg_matches():
    from dory.wsp.lens import ablate, teacher_forced_logprob
    model = DoryTorch.from_config(CFG)
    ids = list(range(1, 13))
    c = teacher_forced_logprob(model, ids, 6, 2)
    assert ablate(model, ids, 6, 2, "layer.2", clean=c) == ablate(model, ids, 6, 2, "layer.2")


def test_ablation_resume(tmp_path):
    model = DoryTorch.from_config(CFG)
    rng = np.random.default_rng(1)
    mk = lambda: (list(rng.integers(1, 64, 12)), 6)
    groups = {"trunc": [mk(), mk()], "fin": [mk(), mk()]}
    full = wc.run_ablations(model, groups, top_attn=1, top_mlp=1, n_neurons=2)
    part = {k: v for k, v in full.items() if k.startswith("layer.")}
    calls = []
    wc.run_ablations(model, groups, top_attn=1, top_mlp=1, n_neurons=2, res=part, checkpoint=lambda r: calls.append(len(r)))
    assert part == full and calls and calls[-1] == len(full)
