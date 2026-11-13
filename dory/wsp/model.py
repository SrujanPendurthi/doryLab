"""PyTorch Dory (transformers PR #50099 `modeling_dory.py`) wrapped for activation capture.

Loop count: `DoryModel.forward` reads `config.n_recurrent_loops` on every call, so `forward(ids, loops)` sets that
config field before running (no forward kwarg exists). KV cache is off, so cache mode is irrelevant.

Virtual layers are half-blocks as in the checkpoint's `layer_types` (attention, mlp, attention, ...): virtual 2c is
the attention half and 2c+1 the MLP half of the c-th decoder-layer call (HF fuses a pair into one decoder layer
and halves the layer counts; recurrent layers are called once per loop). Total = 16 + 40*loops + 16.
Captures (float32, batch dim dropped):
  resid.{v}     residual stream after half-block v (attn: input of pre_feedforward_layernorm; mlp: layer output)
  attn_out.{v}  attention half only: o_proj input (gated, attn_sub_norm'd heads) as [T, H, D], before o_proj/residual
  mlp_act.{v}   mlp half only: act(gate)*up (input of ffn_sub_norm), [T, intermediate]
"""
import json
from pathlib import Path

import torch
import urllib.request

from transformers import AutoModelForCausalLM, DoryConfig

N_IN, N_REC, N_OUT = 16, 40, 16


def virtual_to_physical(v: int, loops: int) -> tuple[int, int]:
    """Virtual half-block -> (physical half-block, loop index; -1 outside the recurrent block)."""
    if v < N_IN:
        return v, -1
    rec = N_REC * loops
    if v < N_IN + rec:
        return N_IN + (v - N_IN) % N_REC, (v - N_IN) // N_REC
    return v - rec + N_REC, -1


class DoryTorch:
    def __init__(self, model):
        self.model = model.eval()

    @classmethod
    def load(cls, model_dir: Path, device: str = "mps", dtype=torch.bfloat16) -> "DoryTorch":
        return cls(AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=dtype).to(device))

    @classmethod
    def from_config(cls, cfg: dict, seed: int = 0, device: str = "cpu") -> "DoryTorch":
        torch.manual_seed(seed)
        return cls(AutoModelForCausalLM.from_config(DoryConfig(**cfg)).to(device))

    @torch.no_grad()
    def forward(self, ids: torch.LongTensor, loops: int, capture: set[str] | None = None):
        m = self.model
        m.config.n_recurrent_loops = loops
        cap: dict[str, torch.Tensor] = {}
        want = capture or set()
        state = {"c": -1}  # decoder-layer call counter
        hooks = []

        def put(key, t):
            if key in want or key.split(".")[0] in want:
                cap[key] = t[0].detach().float()

        for layer in m.model.layers:
            def pre(mod, args, kwargs):
                state["c"] += 1
            def mid(mod, args):  # residual after attention half
                put(f"resid.{2 * state['c']}", args[0])
            def post(mod, args, out):
                put(f"resid.{2 * state['c'] + 1}", out)
            def attn(mod, args):
                x = args[0]
                put(f"attn_out.{2 * state['c']}", x.view(*x.shape[:-1], -1, mod.parent_head_dim))
            def act(mod, args):
                put(f"mlp_act.{2 * state['c'] + 1}", args[0])
            layer.self_attn.o_proj.parent_head_dim = layer.self_attn.head_dim
            hooks += [layer.register_forward_pre_hook(pre, with_kwargs=True),
                      layer.pre_feedforward_layernorm.register_forward_pre_hook(mid),
                      layer.register_forward_hook(post),
                      layer.self_attn.o_proj.register_forward_pre_hook(attn),
                      layer.mlp.ffn_sub_norm.register_forward_pre_hook(act)]
        try:
            logits = m(input_ids=ids.to(m.device), use_cache=False).logits[0].float()
        finally:
            for h in hooks:
                h.remove()
        return logits, cap


def _post(url: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def template_ids(server_url: str, prompt: str) -> list[int]:
    from dory.client import CHAT_TEMPLATE_KWARGS
    body = {"messages": [{"role": "user", "content": prompt}]}
    if CHAT_TEMPLATE_KWARGS:
        body["chat_template_kwargs"] = CHAT_TEMPLATE_KWARGS
    text = _post(server_url, "/apply-template", body)["prompt"]
    return _post(server_url, "/tokenize", {"content": text, "add_special": False, "parse_special": True})["tokens"]
