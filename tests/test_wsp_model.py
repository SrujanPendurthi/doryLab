import json
import urllib.request
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "DoryConfig"):
    pytest.skip("transformers build without dory (use .venv-wsp)", allow_module_level=True)

from dory.wsp.model import DoryTorch, template_ids, virtual_to_physical

ROOT = Path(__file__).resolve().parent.parent
BF16 = ROOT / "models/bf16"
GGUF = ROOT / "models/gguf/dory-base-bf16.gguf"
needs_model = pytest.mark.skipif(not (BF16.exists() and GGUF.exists()), reason="models/bf16 or bf16 gguf absent")


def test_virtual_to_physical():
    assert virtual_to_physical(0, 2) == (0, -1)
    assert virtual_to_physical(16, 2) == (16, 0)
    assert virtual_to_physical(56, 2) == (16, 1)
    assert virtual_to_physical(96, 2) == (56, -1)


def test_capture_counts():
    cfg = dict(vocab_size=64, hidden_size=32, num_hidden_layers=6, intermediate_size=48, num_attention_heads=2,
               num_key_value_heads=1, head_dim=16, n_input_layers=2, n_recurrent_layers=2, n_output_layers=2,
               n_recurrent_loops=2, layer_types=["full_attention", "mlp"] * 3, pad_token_id=0)
    m = DoryTorch.from_config(cfg)
    logits, cap = m.forward(torch.arange(5)[None], loops=3, capture={"resid", "attn_out", "mlp_act"})
    assert logits.shape == (5, 64)
    assert len([k for k in cap if k.startswith("resid.")]) == 2 + 2 * 3 + 2
    assert cap["attn_out.0"].shape == (5, 2, 16)
    assert cap["mlp_act.1"].shape == (5, 48)


def _post(url, body):
    req = urllib.request.Request(url + "/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


@pytest.fixture(scope="module")
def parity():
    """Phase 1: BF16 GGUF server (one at a time) gives references; phase 2: torch model (fp32, so the two
    never sit in memory together; bf16 torch activations alone drift ~0.2 logprob). The server runs on CPU (-ngl 0):
    the Metal backend's lower-precision activations drift from fp32 torch by up to 0.3 logprob at k=4, while the CPU
    backend agrees with torch within ~0.035 (see task-5 report)."""
    from dory.evalsets import load_split
    from dory.server import Server, free_port, server_args
    prompts = [it.prompt for it in load_split("train")[:20]]
    ids, ref = [], {}
    for k in (1, 2, 4):
        port = free_port()
        srv = Server(server_args(GGUF, k, port, slots=1, extra=["-ngl", "0"]), port, startup_timeout=600)
        try:
            if k == 1:
                ids = [template_ids(srv.url, p) for p in prompts]
            for i, t in enumerate(ids):
                d = _post(srv.url, {"prompt": t, "n_predict": 1, "n_probs": 5, "temperature": 0, "cache_prompt": False})
                ref[k, i] = d["completion_probabilities"][0]
            if k == 1:
                for i, t in enumerate(ids):
                    d = _post(srv.url, {"prompt": t, "n_predict": 32, "temperature": 0, "return_tokens": True,
                                        "cache_prompt": False})
                    ref["greedy", i] = d["tokens"]
        finally:
            srv.stop()
    return ids, ref, DoryTorch.load(BF16, dtype=torch.float32)


@needs_model
@pytest.mark.parametrize("k", [1, 2, 4])
def test_parity_next_token(parity, k):
    ids, ref, model = parity
    bad, worst = [], 0.0
    for i, t in enumerate(ids):
        top = ref[k, i]
        lp = torch.log_softmax(model.forward(torch.tensor([t]), loops=k)[0][-1].cpu(), -1)
        if int(lp.argmax()) != top["id"]:
            bad.append(i)
        worst = max([worst] + [abs(float(lp[x["id"]]) - x["logprob"]) for x in top["top_logprobs"]])
    assert not bad, f"top-1 mismatch on prompts {bad}"
    assert worst < 0.1, worst


@needs_model
def test_parity_greedy_32(parity):
    ids, ref, model = parity
    model.model.config.n_recurrent_loops = 1
    same = 0
    for i, t in enumerate(ids):
        out = model.model.generate(torch.tensor([t], device=model.model.device), max_new_tokens=32, do_sample=False)
        same += out[0, len(t):].tolist() == ref["greedy", i]
    assert same >= 18, same
