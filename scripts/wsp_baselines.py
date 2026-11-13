"""WSP Task 7 Steps 1-2: base Q8_0 baselines, determinism check, noise floor, research rows.
Run under .venv-wsp (to_gguf needs the PR transformers): PYTHONPATH=. .venv-wsp/bin/python -m scripts.wsp_baselines
Order puts what the edit loop needs first; holdout k=2/4 (stretch goal only) run last."""
import json
import random
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open

from dory import evalsets
from dory.sweep import read_rows, run_sweep
from dory.wsp.edits import apply_edit, to_gguf
from dory.wsp.run import CACHE, evaluate, sha256_file, summarize

BASE = Path("models/gguf/dory-base-q8_0.gguf")
BF16 = Path("models/bf16")
OUT = Path("results/wsp/baselines.json")
FRAC = 0.00016  # 0.016% of recurrent MLP down-proj params, as in the Sparks edit size
DOWN = [f"backbone.layers.{L}.mixer.down_proj.weight" for L in range(17, 56, 2)]


def log(stage, **kw):
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    res[stage] = kw
    OUT.write_text(json.dumps(res, indent=1))
    print(stage, json.dumps(kw)[:300], flush=True)


def noise_edit(seed: int) -> dict:
    rng = random.Random(seed)
    g = torch.Generator().manual_seed(seed)
    ops = []
    with safe_open(str(BF16 / "model.safetensors"), "pt") as f:
        sizes = {n: f.get_slice(n).get_shape() for n in DOWN}
        total = sum(r * c for r, c in sizes.values())
        n_per = round(total * FRAC / len(DOWN))
        for n, (r, c) in sizes.items():
            std = f.get_tensor(n).float().std().item()
            idx = {(rng.randrange(r), rng.randrange(c)) for _ in range(n_per)}
            idx = sorted(idx)
            vals = (torch.randn(len(idx), generator=g) * std).tolist()
            ops.append({"tensor": n, "op": "add", "index": [list(i) for i in idx], "value": vals})
    return {"id": f"noise{seed}", "hypothesis": "noise floor", "evidence": [], "ops": ops}


def main():
    sha = sha256_file(BASE)
    d = CACHE / sha[:12]
    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    OUT.parent.mkdir(parents=True, exist_ok=True)

    if "screen_k1" not in done:
        log("screen_k1", **summarize(evaluate(BASE, "wsp_screen", 1)))
    if "determinism" not in done:
        items = evalsets.load_split("wsp_screen")
        det = d / "wsp_screen_k1_det.jsonl"
        run_sweep(items, [1], "wsp_det", BASE, det, slots=4)
        a = {r["id"]: r["output"] for r in evaluate(BASE, "wsp_screen", 1)}
        b = {r["id"]: r["output"] for r in read_rows(det) if r["variant"] == "wsp_det"}
        same = sum(a[i] == b.get(i) for i in a)
        log("determinism", identical=same, n=len(a), slots=4, ok=same >= 57)
        if same < 57:
            sys.exit("nondeterministic at slots=4: switch evaluate default to slots=1 and clear cache")
    for seed in (0, 1, 2):
        key = f"noise{seed}"
        if key in done:
            continue
        mdir, g = Path(f"results/wsp/models/{key}"), Path(f"results/wsp/models/{key}.gguf")
        st = apply_edit(noise_edit(seed), BF16, mdir)
        to_gguf(mdir, g)
        s = summarize(evaluate(g, "wsp_screen", 1))
        log(key, n_changed=st.n_changed, n_requested=st.n_requested, **s)
        shutil.rmtree(mdir)
        g.unlink()
    for split, k in (("holdout", 1), ("arc", 1)):
        if f"{split}_k{k}" not in done:
            log(f"{split}_k{k}", **summarize(evaluate(BASE, split, k)))
    if "research_k1" not in done:
        tr = evalsets.load_split("train")
        items = [it for s in ("gsm8k", "mmlu_pro", "mbpp") for it in [i for i in tr if i.source == s][:100]]
        out = d / "research_k1.jsonl"
        run_sweep(items, [1], "wsp", BASE, out, slots=4)
        rows = [r for r in read_rows(out) if r["variant"] == "wsp"]
        log("research_k1", **summarize(rows))
    for k in (2, 4):
        if f"holdout_k{k}" not in done:
            log(f"holdout_k{k}", **summarize(evaluate(BASE, "holdout", k)))
    print("WSP_BASELINES_DONE", flush=True)


if __name__ == "__main__":
    main()
