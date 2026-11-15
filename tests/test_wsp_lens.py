import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "DoryConfig"):
    pytest.skip("transformers build without dory (use .venv-wsp)", allow_module_level=True)

from dory.evalsets import Item
from dory.wsp.lens import ablate, collect_resid, logit_lens, permute_options, teacher_forced_logprob, train_probe
from dory.wsp.model import DoryTorch

CFG = dict(vocab_size=64, hidden_size=64, num_hidden_layers=6, intermediate_size=96, num_attention_heads=4,
           num_key_value_heads=2, head_dim=16, n_input_layers=2, n_recurrent_layers=2, n_output_layers=2,
           n_recurrent_loops=2, layer_types=["full_attention", "mlp"] * 3, pad_token_id=0)
IDS = list(range(1, 13))
LOOPS = 2


@pytest.fixture(scope="module")
def model():
    return DoryTorch.from_config(CFG)


def test_forward_restores_loops(model):
    model.forward(torch.tensor([IDS]), loops=3)
    assert model.model.config.n_recurrent_loops == 2


def test_lens_final_layer_equals_logits(model):
    logits, _ = model.forward(torch.tensor([IDS]), LOOPS)
    lens = logit_lens(model, IDS, LOOPS, topk=5, target_ids=IDS[1:] + [0])
    assert lens["top_ids"].shape == (2 + 2 * LOOPS + 2, len(IDS), 5)
    assert lens["target_rank"].shape == (2 + 2 * LOOPS + 2, len(IDS))
    np.testing.assert_array_equal(lens["top_ids"][-1], logits.topk(5, -1).indices.numpy())


@pytest.mark.parametrize("comp", ["layer.2", "layer.3", "mlp.3", "head.2.1", "neurons.3:0,5,7", "layer.1"])
@pytest.mark.parametrize("mode", ["zero", "mean"])
def test_ablate_zero_layer_changes_logprob(model, comp, mode):
    d = ablate(model, IDS, 6, LOOPS, comp, mode)
    assert np.isfinite(d) and d != 0.0
    assert ablate(model, IDS, 6, LOOPS, comp, mode) == d  # hooks removed, deterministic
    assert len(teacher_forced_logprob(model, IDS, 6, LOOPS)) == 6


def test_ablate_applies_in_every_loop(model):
    assert ablate(model, IDS, 6, 1, "layer.2") != ablate(model, IDS, 6, 3, "layer.2")


@pytest.mark.parametrize("comp", ["bogus.1", "mlp.2", "head.3.0", "layer.99", "head.2", "neurons.3"])
def test_ablate_unknown_component_raises(model, comp):
    with pytest.raises(ValueError):
        ablate(model, IDS, 6, LOOPS, comp, "zero")


def test_collect_resid(model):
    assert collect_resid(model, [IDS, IDS[:5]], LOOPS, [0, 3], "prompt_end").shape == (2, 2, 64)
    assert collect_resid(model, [IDS, IDS[:5]], LOOPS, [0, 3], "all").shape == (17, 2, 64)


def test_probe_shuffled_control_near_chance():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    X = rng.normal(size=(200, 16))
    X[:, 0] += 3 * y
    r = train_probe(X, y)
    assert r["auc_mean"] > 0.9 and 0.35 <= r["shuf_auc_mean"] <= 0.65


def test_permute_options_remaps_gold():
    it = Item("mmlu_pro:x:0", "mmlu_pro", "Q?\nA. a\nB. b\nC. c\nD. d\n\nEnd with 'The answer is (X)'.", "B")
    out = permute_options(it, [2, 0, 3, 1])
    assert out.prompt == "Q?\nA. c\nB. a\nC. d\nD. b\n\nEnd with 'The answer is (X)'."
    assert out.gold == "D" and permute_options(it, [1, 0, 2, 3]).gold == "A"
    with pytest.raises(ValueError):
        permute_options(it, [0, 1])


def test_lens_positions_matches_slice(model):
    tgt = IDS[1:] + [0]
    full = logit_lens(model, IDS, LOOPS, topk=3, target_ids=tgt)
    sub = logit_lens(model, IDS, LOOPS, topk=3, target_ids=tgt, positions=[4, 7])
    np.testing.assert_array_equal(sub["target_rank"], full["target_rank"][:, [4, 7]])
    np.testing.assert_array_equal(sub["top_ids"], full["top_ids"][:, [4, 7]])
