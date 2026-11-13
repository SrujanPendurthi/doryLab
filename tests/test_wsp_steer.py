import numpy as np
import pytest
import torch

pytest.importorskip("transformers.models.dory")
from dory.wsp.model import DoryTorch  # noqa: E402
from scripts.wsp_steer import hooks_on  # noqa: E402
from tests.test_wsp_corpus import CFG  # noqa: E402


def test_hook_equals_weight_edit():
    """The precheck hook and the rank-1 weight edit W - a d d^T W give the same logits."""
    m = DoryTorch.from_config(CFG)
    ids = torch.tensor([[1, 5, 9, 3, 7]])
    hid = m.model.config.hidden_size
    p = 2 * m.model.config.n_input_layers + 1  # first recurrent MLP half
    d = np.random.default_rng(0).standard_normal(hid)
    d /= np.linalg.norm(d)
    a = 0.7
    hs = hooks_on(m, {p: d}, a)
    hooked, _ = m.forward(ids, 2)
    [h.remove() for h in hs]
    W = m.model.model.layers[p // 2].mlp.down_proj.weight
    dt = torch.tensor(d, dtype=W.dtype)
    with torch.no_grad():
        W -= a * dt[:, None] @ (dt[None, :] @ W)
    edited, _ = m.forward(ids, 2)
    assert torch.allclose(hooked, edited, atol=1e-4)
