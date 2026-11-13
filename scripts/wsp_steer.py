"""WSP Task 8: direction-ablation edits on recurrent down_proj (controller tool).

  precheck: PYTHONPATH=. .venv-wsp/bin/python -m scripts.wsp_steer precheck --dir trunc --alphas 0.5 1 --layers 17-55
  write:    PYTHONPATH=. .venv-wsp/bin/python -m scripts.wsp_steer write --dir trunc --alpha 1 --layers 17-55 --id e001

Edit: for each MLP half p, W_p' = W_p - alpha * d_p d_p^T W_p (removes alpha of d_p from that half's write; alpha=1 is a
full projection). Precheck applies the identical map as a forward hook on down_proj output and reports mean change in
teacher-forced logprob per token of the model's own continuation, per group, against a random-direction control.
"""
import argparse
import json

import numpy as np
import torch
from safetensors import safe_open

from dory import evalsets
from scripts.wsp_corpus import BF16, ROOT, get_tok, is_trunc, load_research_rows, row_ids

DIRS = ROOT / "results/wsp/corpus/directions.npz"
EDITS = ROOT / "research/wsp/edits"


def directions(kind: str) -> dict[int, np.ndarray]:
    z = np.load(DIRS)
    L, src, tr, ok = list(z["layers"]), z["source"], z["trunc"], z["correct"]
    if kind == "trunc":  # within MMLU-Pro: truncated minus finished, mean generated-token residual
        m = src == "mmlu_pro"
        d = z["tokmean"][m & tr].mean(0) - z["tokmean"][m & ~tr].mean(0)
    elif kind == "commit":  # commit position minus row mean, finished gsm8k/mmlu rows with a commit position
        c = z["commit"]
        m = ~np.isnan(c[:, 0, 0]) & ~tr
        d = (c[m] - z["tokmean"][m]).mean(0)
    else:
        raise ValueError(kind)
    return {p: d[i] / np.linalg.norm(d[i]) for i, p in enumerate(L)}


def parse_layers(s: str) -> list[int]:
    a, b = map(int, s.split("-"))
    return [p for p in range(a, b + 1) if p % 2 == 1]


def hooks_on(model, dirs: dict[int, np.ndarray], alpha: float):
    hs = []
    for p, d in dirs.items():
        mod = model.model.model.layers[p // 2].mlp.down_proj
        dt = torch.tensor(d, dtype=torch.float32)

        def f(_m, _i, out, dt=dt):
            v = dt.to(out.device, out.dtype)
            return out - alpha * (out @ v)[..., None] * v
        hs.append(mod.register_forward_hook(f))
    return hs


def groups(n=12):
    rows = load_research_rows()
    pick = lambda f: [r for r in rows if f(r)][:n]
    return {"mmlu_trunc": pick(lambda r: r["source"] == "mmlu_pro" and is_trunc(r)),
            "mmlu_ok": pick(lambda r: r["source"] == "mmlu_pro" and r["correct"] and not is_trunc(r)),
            "gsm8k_ok": pick(lambda r: r["source"] == "gsm8k" and r["correct"] and not is_trunc(r)),
            "mbpp_ok": pick(lambda r: r["source"] == "mbpp" and r["correct"])}


def precheck(kind, alphas, layers, n=12, seed=0):
    from dory.wsp.lens import teacher_forced_logprob
    from dory.wsp.model import DoryTorch
    tok, items = get_tok(), {it.id: it for it in evalsets.load_split("train")}
    model = DoryTorch.load(BF16)
    g = {k: [row_ids(tok, r, items[r["id"]]) for r in v] for k, v in groups(n).items()}
    lp = lambda: {k: [teacher_forced_logprob(model, ids, npr, 1) for ids, npr in v] for k, v in g.items()}
    with torch.no_grad():
        base = lp()
        real = {p: d for p, d in directions(kind).items() if p in layers}
        rng = np.random.default_rng(seed)
        rand = {p: (lambda x: x / np.linalg.norm(x))(rng.standard_normal(len(d))) for p, d in real.items()}
        out = {}
        for a in alphas:
            for name, dd in (("dir", real), ("rand", rand)):
                hs = hooks_on(model, dd, a)
                try:
                    cur = lp()
                finally:
                    [h.remove() for h in hs]
                out[f"{name}@{a}"] = {k: float(np.mean([(c - b).mean() for c, b in zip(cur[k], base[k])])) for k in g}
                print(f"{name}@{a}", json.dumps({k: round(v, 4) for k, v in out[f'{name}@{a}'].items()}), flush=True)
    return out


def write_edit(kind, alpha, layers, eid, hypothesis, evidence, predict):
    dirs = {p: d for p, d in directions(kind).items() if p in layers}
    arrs, ops = {}, []
    with safe_open(str(BF16 / "model.safetensors"), "pt") as f:
        for p, d in dirs.items():
            name = f"backbone.layers.{p}.mixer.down_proj.weight"
            W = f.get_tensor(name).float().numpy()  # [hidden, inter]
            arrs[f"u{p}"] = (-alpha * d)[:, None].astype(np.float32)
            arrs[f"v{p}"] = (W.T @ d)[:, None].astype(np.float32)
            ops.append({"tensor": name, "op": "lowrank_add", "npz": str((EDITS / f"{eid}.npz").relative_to(ROOT)),
                        "u": f"u{p}", "v": f"v{p}"})
    EDITS.mkdir(parents=True, exist_ok=True)
    np.savez(EDITS / f"{eid}.npz", **arrs)
    edit = {"id": eid, "hypothesis": hypothesis, "evidence": evidence, "predict": predict, "ops": ops,
            "recipe": {"dir": kind, "alpha": alpha, "layers": layers}}
    (EDITS / f"{eid}.json").write_text(json.dumps(edit, indent=1))
    print(EDITS / f"{eid}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["precheck", "write"])
    ap.add_argument("--dir", default="trunc")
    ap.add_argument("--alphas", type=float, nargs="*", default=[0.5, 1.0])
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--layers", default="17-55")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--id")
    ap.add_argument("--hypothesis", default="")
    ap.add_argument("--evidence", nargs="*", default=[])
    ap.add_argument("--predict", default="")
    a = ap.parse_args()
    L = parse_layers(a.layers)
    if a.cmd == "precheck":
        res = precheck(a.dir, a.alphas, L, a.n)
        p = ROOT / f"results/wsp/corpus/precheck_{a.dir}_{a.layers}.json"
        p.write_text(json.dumps(res, indent=1))
    else:
        write_edit(a.dir, a.alpha, L, a.id, a.hypothesis, a.evidence, a.predict)


if __name__ == "__main__":
    main()
