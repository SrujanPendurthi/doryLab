"""WSP Task 8 research: residual directions for edit design (controller tool).
Run: PYTHONPATH=. .venv-wsp/bin/python -m scripts.wsp_directions
Per research row at loops=1: mean residual over generated tokens, and the residual at the position that predicts the
first token of the last "answer is" phrase sentence ("commit" position), at every recurrent MLP half (17..55 odd).
Writes results/wsp/corpus/directions.npz: tokmean [N,L,D], commit [N,L,D] (nan if none), ids, source, trunc, correct, layers.
"""
import json

import numpy as np
import torch

from dory import evalsets
from dory.wsp.model import DoryTorch
from scripts.wsp_corpus import BF16, ROOT, char_to_token, get_tok, is_trunc, load_research_rows, pieces_of, row_ids

OUT = ROOT / "results/wsp/corpus/directions.npz"
LAYERS = list(range(17, 56, 2))


def commit_index(pieces) -> int | None:
    """Output-token index of the token that starts the sentence holding the last 'answer is' (prediction pos = idx-1)."""
    text = "".join(pieces)
    k = text.lower().rfind("answer is")
    if k < 0:
        return None
    s = max(text.rfind(c, 0, k) for c in ".\n*:") + 1  # sentence start
    while s < k and text[s] in " *\n":
        s += 1
    return char_to_token(pieces, s)


def main():
    rows, tok = load_research_rows(), get_tok()
    items = {it.id: it for it in evalsets.load_split("train")}
    model = DoryTorch.load(BF16)
    keys = {f"resid.{v}" for v in LAYERS}
    tm, cm = [], []
    for n, r in enumerate(rows):
        ids, npr = row_ids(tok, r, items[r["id"]])
        with torch.no_grad():
            _, acts = model.forward(torch.tensor([ids]), 1, keys)
        R = torch.stack([acts[f"resid.{v}"] for v in LAYERS], 1).cpu().numpy()  # [T, L, D]
        tm.append(R[npr:].mean(0))
        ci = commit_index(pieces_of(tok, ids[npr:])) if r["source"] != "mbpp" else None
        cm.append(R[npr + ci - 1] if ci else np.full_like(tm[-1], np.nan))
        if n % 25 == 0:
            print(n, flush=True)
    np.savez(OUT, tokmean=np.stack(tm), commit=np.stack(cm), layers=np.array(LAYERS),
             ids=np.array([r["id"] for r in rows]), source=np.array([r["source"] for r in rows]),
             trunc=np.array([is_trunc(r) for r in rows]), correct=np.array([bool(r["correct"]) for r in rows]))
    print("DIRECTIONS_DONE", flush=True)


if __name__ == "__main__":
    main()
