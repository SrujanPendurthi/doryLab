"""Instruments over `DoryTorch`: logit lens, teacher-forced logprob, component ablation, residual probes, option permutation.

Layer numbers: `layers`/lens rows are virtual half-blocks (see model.py); ablation components use the physical
half-block index (0..71, even = attention half, odd = MLP half; decoder layer = phys // 2). Hooks sit on the module,
so a recurrent physical layer is ablated in every loop.
Note the between-loop re-injection is not captured (see model.py): lens rows at v = 16+40j (j>=1) follow a jump.
"""
import dataclasses
import re

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from dory.evalsets import Item


def _ids(ids) -> torch.LongTensor:
    return torch.as_tensor(ids, dtype=torch.long).reshape(1, -1)


@torch.no_grad()
def logit_lens(model, ids, loops: int, topk: int = 5, target_ids=None) -> dict:
    """Final norm + lm_head on every `resid.{v}`. target_ids: [T] token per position (e.g. next token) -> target_rank
    (0 = top-1)."""
    m = model.model
    _, acts = model.forward(_ids(ids), loops, {"resid"})
    dt = next(m.parameters()).dtype
    tgt = None if target_ids is None else torch.as_tensor(target_ids).reshape(-1).to(m.device)
    top_ids, top_p, rank = [], [], []
    for v in range(len(acts)):
        lg = m.lm_head(m.model.norm(acts[f"resid.{v}"].to(dtype=dt, device=m.device))).float()
        p, i = lg.softmax(-1).topk(topk, -1)
        top_ids.append(i.cpu()), top_p.append(p.cpu())
        if tgt is not None:
            rank.append((lg > lg.gather(1, tgt[:, None])).sum(-1).cpu())
    out = {"top_ids": torch.stack(top_ids).numpy(), "top_p": torch.stack(top_p).numpy()}
    if tgt is not None:
        out["target_rank"] = torch.stack(rank).numpy()
    return out


def teacher_forced_logprob(model, ids, n_prompt: int, loops: int) -> np.ndarray:
    """Logprob of each token ids[n_prompt:] given its prefix."""
    x = _ids(ids)
    logits, _ = model.forward(x, loops)
    lp = logits[n_prompt - 1:-1].log_softmax(-1)
    return lp.gather(1, x[0, n_prompt:, None].to(lp.device))[:, 0].cpu().numpy()


def _parse(model, component: str):
    layers = model.model.model.layers
    kind, _, rest = component.partition(".")
    try:
        if kind == "neurons":
            phys, idx = rest.split(":")
            args = int(phys), [int(i) for i in idx.split(",")]
        elif kind == "head":
            phys, h = rest.split(".")
            args = int(phys), int(h)
        elif kind in ("mlp", "layer"):
            args = int(rest), None
        else:
            raise ValueError
    except ValueError:
        raise ValueError(f"bad component {component!r}") from None
    phys = args[0]
    if not 0 <= phys < 2 * len(layers):
        raise ValueError(f"physical layer out of range in {component!r}")
    if (kind in ("mlp", "neurons") and phys % 2 == 0) or (kind == "head" and phys % 2 == 1):
        raise ValueError(f"{component!r}: physical {phys} is the wrong half-block type")
    return kind, layers[phys // 2], phys % 2 == 0, args[1]


def _replace(x, mode):
    return torch.zeros_like(x) if mode == "zero" else x.mean(1, keepdim=True).expand_as(x).clone()


def ablate(model, ids, n_prompt: int, loops: int, component: str, mode: str = "zero") -> float:
    """Mean (ablated - clean) teacher-forced logprob over ids[n_prompt:]. `ids` should hold the model's own continuation.

    zero: head/neurons are set to 0; mlp/layer are skipped (the half-block leaves the residual unchanged, since the
    nGPT update is h + rate*(norm(out) - h) and out is replaced by the unit-norm input h).
    mean: head/neurons/sublayer output replaced by its mean over token positions in the current run (includes
    future positions, a standard mean-ablation approximation); mlp/layer mean replaces the pre-norm sublayer output.
    """
    if mode not in ("zero", "mean"):
        raise ValueError(mode)
    kind, layer, is_attn, arg = _parse(model, component)
    clean = teacher_forced_logprob(model, ids, n_prompt, loops)
    if kind == "head":
        d = layer.self_attn.head_dim

        def fn(mod, args):
            x = args[0].clone()
            x[..., arg * d:(arg + 1) * d] = _replace(x, mode)[..., arg * d:(arg + 1) * d]
            return (x,)
        h = layer.self_attn.o_proj.register_forward_pre_hook(fn)
    elif kind == "neurons":
        def fn(mod, args):
            x = args[0].clone()
            x[..., arg] = _replace(x, mode)[..., arg]
            return (x,)
        h = layer.mlp.down_proj.register_forward_pre_hook(fn)
    else:
        sub = layer.self_attn if is_attn else layer.mlp

        def fn(mod, args, kwargs, out):
            if mode == "mean":
                return (_replace(out[0], mode), out[1]) if is_attn else _replace(out, mode)
            x = args[0] if args else kwargs["hidden_states"]
            return (x, out[1]) if is_attn else x
        h = sub.register_forward_hook(fn, with_kwargs=True)
    try:
        abl = teacher_forced_logprob(model, ids, n_prompt, loops)
    finally:
        h.remove()
    return float((abl - clean).mean())


def collect_resid(model, items_ids, loops: int, layers: list[int], position: str = "prompt_end") -> np.ndarray:
    """Residuals at virtual `layers`: [N, len(layers), D] (prompt_end = last token) or [sum T, len(layers), D] (all)."""
    if position not in ("prompt_end", "all"):
        raise ValueError(position)
    keys = {f"resid.{v}" for v in layers}
    rows = []
    for ids in items_ids:
        _, acts = model.forward(_ids(ids), loops, keys)
        r = torch.stack([acts[f"resid.{v}"] for v in layers], 1)  # [T, L, D]
        rows.append((r[-1:] if position == "prompt_end" else r).cpu())
    return torch.cat(rows).numpy()


def train_probe(X, y, seed: int = 0) -> dict:
    """Logistic probe, 5-fold CV ROC-AUC; `shuf_*` repeats it with permuted labels as a control."""
    X, y = np.asarray(X), np.asarray(y)
    scoring = "roc_auc" if len(np.unique(y)) == 2 else "roc_auc_ovr"

    def cv(labels):
        s = cross_val_score(LogisticRegression(max_iter=1000), X, labels, scoring=scoring,
                            cv=StratifiedKFold(5, shuffle=True, random_state=seed))
        return float(s.mean()), float(s.std())
    (auc_mean, auc_std), (shuf_mean, shuf_std) = cv(y), cv(np.random.default_rng(seed).permutation(y))
    return {"auc_mean": auc_mean, "auc_std": auc_std, "shuf_auc_mean": shuf_mean, "shuf_auc_std": shuf_std}


def permute_options(item: Item, perm: list[int]) -> Item:
    """New option j is old option perm[j]; letters are rewritten in order and gold is remapped."""
    if item.source not in ("mmlu_pro", "arc"):
        raise ValueError(f"no options in source {item.source!r}")
    lines = item.prompt.split("\n")
    start = max(i for i, l in enumerate(lines) if l.startswith("A. "))
    n = 0
    while start + n < len(lines) and re.match(rf"{chr(65 + n)}\. ", lines[start + n]):
        n += 1
    if sorted(perm) != list(range(n)):
        raise ValueError(f"perm must permute 0..{n - 1}")
    opts = [lines[start + i][3:] for i in range(n)]
    lines[start:start + n] = [f"{chr(65 + j)}. {opts[p]}" for j, p in enumerate(perm)]
    return dataclasses.replace(item, prompt="\n".join(lines), gold=chr(65 + perm.index(ord(item.gold) - 65)))
