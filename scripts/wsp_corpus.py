"""WSP Task 7 Step 3: research-corpus numbers. Each stage is idempotent (skips if its output exists).
Run under .venv-wsp: PYTHONPATH=. .venv-wsp/bin/python -m scripts.wsp_corpus --stage baselines|taxonomy|lens|probes|ablations|fp32check
Outputs: <out-root>/results/wsp/corpus/*.json|npz and <out-root>/research/wsp/corpus/0N-*.md.
Rank convention: lens target_rank 0 = top-1 (the brief's "rank == 1").
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from dory import evalsets
from dory.wsp.run import CACHE, sha256_file

ROOT = Path(__file__).resolve().parents[1]
BASE_GGUF = ROOT / "models/gguf/dory-base-q8_0.gguf"
BF16 = ROOT / "models/bf16"
NGRAM = 8
PHRASE = {"gsm8k": "answer is", "mmlu_pro": "answer is"}


# ---------- io helpers ----------
def dirs(out_root: Path):
    j, m = out_root / "results/wsp/corpus", out_root / "research/wsp/corpus"
    j.mkdir(parents=True, exist_ok=True), m.mkdir(parents=True, exist_ok=True)
    return j, m


def load_research_rows(path: Path | None = None) -> list[dict]:
    path = path or CACHE / sha256_file(BASE_GGUF)[:12] / "research_k1.jsonl"
    rows = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get("variant") == "wsp":
            rows[r["id"]] = r
    return list(rows.values())


def table(header, body) -> str:
    f = lambda c: f"{c:.3f}" if isinstance(c, float) else str(c)
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                     + ["| " + " | ".join(f(c) for c in r) + " |" for r in body]) + "\n"


def is_trunc(r) -> bool:
    return not r.get("error") and r["n_gen"] >= evalsets.MAX_TOKENS[r["source"]]


# ---------- stage 1: baselines ----------
def stage_baselines(out_root: Path, baselines: Path | None = None):
    _, md = dirs(out_root)
    out = md / "00-baselines.md"
    if out.exists():
        return print("skip baselines")
    b = json.loads((baselines or out_root / "results/wsp/baselines.json").read_text())
    summ = {k: v for k, v in b.items() if isinstance(v, dict) and "acc" in v}
    srcs = sorted({s for v in summ.values() for s in v["per_source"]})
    rows = [[k, v["n"], v["acc"], v["mean_tokens"], v["trunc_rate"]]
            + [x for s in srcs for x in ((v["per_source"][s]["acc"], v["per_source"][s]["mean_tokens"])
                                         if s in v["per_source"] else ("-", "-"))] for k, v in summ.items()]
    head = ["run", "n", "acc", "mean tokens", "trunc rate"] + [f"{s} {m}" for s in srcs for m in ("acc", "tok")]
    text = "# Baselines (base Q8_0)\n\n" + table(head, rows)
    if "screen_k1" in summ:
        s0 = summ["screen_k1"]
        nz = [[k, v["acc"], v["acc"] - s0["acc"], v["mean_tokens"], v["mean_tokens"] - s0["mean_tokens"],
               v.get("n_changed", "-")] for k, v in summ.items() if k.startswith("noise")]
        text += "\n## Noise floor (random edits vs base screen)\n\n" + table(
            ["run", "screen acc", "d acc", "mean tokens", "d tokens", "n changed"], nz)
    if "determinism" in b:
        text += f"\n## Determinism\n\n`{json.dumps(b['determinism'])}`\n"
    out.write_text(text)


# ---------- stage 2: taxonomy ----------
def word_spans(text):
    return [(m.start(), m.group()) for m in re.finditer(r"\S+", text)]


def classify(text: str, source: str):
    """-> (class, repeated 8-gram or None, first-occurrence char offset or None). Rule order: loop, overlong, other."""
    ws = word_spans(text)
    grams = [tuple(w for _, w in ws[i:i + NGRAM]) for i in range(len(ws) - NGRAM + 1)]
    cnt = Counter(grams)
    first = {}
    for i, g in enumerate(grams):
        first.setdefault(g, i)
    rep = [g for g, c in cnt.items() if c >= 3]
    if rep:
        g = min(rep, key=first.get)
        return "loop", " ".join(g), ws[first[g]][0]
    done = text.count("```") >= 2 if source == "mbpp" else PHRASE[source] in text
    return ("other" if done else "overlong"), None, None


def stage_taxonomy(rows, out_root: Path):
    j, md = dirs(out_root)
    if (j / "taxonomy.json").exists():
        return print("skip taxonomy")
    res = []
    for r in rows:
        if is_trunc(r):
            c, g, off = classify(r["output"], r["source"])
            res.append({"id": r["id"], "source": r["source"], "class": c, "ngram": g, "offset": off,
                        "excerpt": r["output"][-300:]})
    (j / "taxonomy.json").write_text(json.dumps(res, indent=1))
    srcs = sorted({r["source"] for r in rows})
    classes = ["loop", "overlong", "other"]
    cnt = [[s] + [sum(x["source"] == s and x["class"] == c for x in res) for c in classes]
           + [sum(1 for r in rows if r["source"] == s)] for s in srcs]
    text = ("# Failure taxonomy (truncated research rows)\n\nEvidence: `results/wsp/corpus/taxonomy.json`. "
            "Rules in order: loop = some 8-gram of whitespace tokens occurs >= 3 times; overlong = no loop and no "
            "answer phrase (`answer is`; mbpp: closing fence); other.\n\n"
            + table(["source"] + classes + ["rows total"], cnt))
    text += "\n## Loop rows\n\n" + table(["id", "8-gram", "first char offset"],
                                         [[x["id"], "`" + x["ngram"].replace("|", "/") + "`", x["offset"]]
                                          for x in res if x["class"] == "loop"])
    for c in classes:
        text += f"\n## Excerpts: {c}\n"
        for x in [x for x in res if x["class"] == c][:3]:
            text += f"\n`{x['id']}`:\n\n```\n{x['excerpt']}\n```\n"
    (md / "01-failure-taxonomy.md").write_text(text)


# ---------- tokenization ----------
def get_tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(BF16))


def prompt_ids(tok, prompt: str) -> list[int]:
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                   tokenize=False, enable_thinking=False)
    return tok.encode(text, add_special_tokens=False)


def row_ids(tok, row, item) -> tuple[list[int], int]:
    p = prompt_ids(tok, item.prompt)
    return p + tok.encode(row["output"], add_special_tokens=False), len(p)


def pieces_of(tok, ids) -> list[str]:
    return [tok.decode([i]) for i in ids]


def char_to_token(pieces, offset: int) -> int:
    """Index of the token containing char `offset` of the concatenated pieces."""
    pos = 0
    for i, p in enumerate(pieces):
        pos += len(p)
        if pos > offset:
            return i
    return len(pieces) - 1


# ---------- stage 3: lens ----------
def target_position(source: str, pieces: list[str]):
    """Index in `pieces` (output tokens) of the first token of the final answer, or None. Whitespace-only tokens are skipped."""
    text = "".join(pieces)
    if source == "mbpp":
        k = text.find("```python")
        e = None if k < 0 else k + len("```python")
    else:
        k = text.rfind(PHRASE[source])
        e = None if k < 0 else k + len(PHRASE[source])
    if e is None:
        return None
    start = 0
    for i, p in enumerate(pieces):
        if start >= e and p.strip():
            if source != "mmlu_pro":
                return i
            if re.fullmatch(r"[A-J]", p.strip(" *()$\\{}")):
                return i
            if i > 0 and start - e > 12:
                return None
        start += len(p)
    return None


def pick_lens_rows(rows, tok, items, per_source):
    out, cnt = [], Counter()
    for r in rows:
        if r.get("error") or cnt[r["source"]] >= per_source:
            continue
        ids = tok.encode(r["output"], add_special_tokens=False)
        t = target_position(r["source"], pieces_of(tok, ids))
        if t is not None:
            out.append((r, t))
            cnt[r["source"]] += 1
    return out


def lens_ranks(model, ids, n_prompt, t, loops):
    """Target-token rank (0 = top-1) at every virtual layer, for output token index t."""
    from dory.wsp.lens import logit_lens
    p = n_prompt + t  # absolute position of the target token; predicted from position p-1
    lens = logit_lens(model, ids[:p], loops, topk=1, target_ids=ids[1:p + 1])
    return lens["target_rank"][:, -1]


def stage_lens(rows, items, out_root: Path, model, tok, n=30, loop_set=(1, 2, 4)):
    j, md = dirs(out_root)
    if (j / "lens.npz").exists():
        return print("skip lens")
    by_id = {it.id: it for it in items}
    sel = pick_lens_rows(rows, tok, items, n // 3)
    arrays, rec = {}, []
    for r, t in sel:
        ids, npr = row_ids(tok, r, by_id[r["id"]])
        for L in loop_set:
            rk = lens_ranks(model, ids, npr, t, L)
            arrays[f"{r['id']}|L{L}"] = rk
            hit = np.nonzero(rk == 0)[0]
            rec.append({"id": r["id"], "source": r["source"], "loops": L,
                        "first_top1": int(hit[0]) if len(hit) else -1, "n_layers": len(rk),
                        "at_boundary": [bool(rk[16 + 40 * (c + 1) - 1] == 0) for c in range(L)]})
    np.savez(j / "lens.npz", **arrays)
    text = ("# Logit lens at the final-answer token\n\nEvidence: `results/wsp/corpus/lens.npz` (rank per virtual layer, "
            "key `id|L<loops>`; rank 0 = top-1). Teacher-forced prompt + own output; first_top1 = -1 means never.\n\n")
    body = []
    for s in sorted({x["source"] for x in rec}):
        for L in loop_set:
            g = [x for x in rec if x["source"] == s and x["loops"] == L]
            d = [x["first_top1"] for x in g if x["first_top1"] >= 0]
            body.append([s, L, len(g), len(d), float(np.median(d)) if d else "-", g[0]["n_layers"]])
    text += table(["source", "loops", "items", "reach top-1", "median first top-1 layer", "n virtual layers"], body)
    body = []
    for L in loop_set:
        for c in range(L):
            g = [x["at_boundary"][c] for x in rec if x["loops"] == L]
            body.append([L, c + 1, 16 + 40 * (c + 1) - 1, float(np.mean(g)) if g else "-"])
    text += "\n## Fraction already top-1 at each loop boundary\n\n" + table(
        ["loops", "after loop", "virtual layer", "fraction top-1"], body)
    (md / "02-lens.md").write_text(text)


# ---------- stage 4: probes ----------
def cv_auc(X, y, groups=None, seed=0, n_null=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    X, y = np.asarray(X), np.asarray(y).astype(int)
    if np.bincount(y, minlength=2).min() < 5:
        return None
    cv = StratifiedKFold(5, shuffle=True, random_state=seed) if groups is None else GroupKFold(5)

    def score(lab):
        pipe = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(max_iter=2000))])
        s = cross_val_score(pipe, X, lab, groups=groups, cv=cv, scoring="roc_auc", error_score=np.nan)
        return float(np.nanmean(s)), float(np.nanstd(s))
    rng = np.random.default_rng(seed)
    real = score(y)
    null = [score(rng.permutation(y))[0] for _ in range(n_null)]
    return {"auc_mean": real[0], "auc_std": real[1], "null_mean": float(np.mean(null)), "null_std": float(np.std(null))}


def probe_layers(total=72):
    return sorted(set(range(0, total, 8)) | {total - 1})  # brief says 0..88 but loops=1 has only 72 virtual layers


def stage_probes(rows, items, out_root: Path, model, tok, tax=None):
    from dory.wsp.lens import collect_resid
    j, md = dirs(out_root)
    if (j / "probes.json").exists():
        return print("skip probes")
    by_id = {it.id: it for it in items}
    layers = probe_layers()
    pids = [prompt_ids(tok, by_id[r["id"]].prompt) for r in rows]
    X = collect_resid(model, pids, 1, layers, "prompt_end")
    res = {"layers": layers, "prompt_end": {}, "onset": {}}
    for name, y in (("will_truncate", [is_trunc(r) for r in rows]), ("will_be_correct", [bool(r["correct"]) for r in rows])):
        res["prompt_end"][name] = {v: cv_auc(X[:, i], y) for i, v in enumerate(layers)}
    tax = tax or json.loads((j / "taxonomy.json").read_text())
    loop_rows = [(r, next(x for x in tax if x["id"] == r["id"])) for r in rows
                 if any(x["id"] == r["id"] and x["class"] == "loop" for x in tax)]
    ol = [layers[-1] if v == "last" else v for v in (40, "last")]  # brief's layer 72 does not exist at loops=1
    feats, labels, groups = [], [], []
    for r, x in loop_rows:
        ids, npr = row_ids(tok, r, by_id[r["id"]])
        pcs = pieces_of(tok, ids[npr:])
        onset = char_to_token(pcs, x["offset"])
        R = collect_resid(model, [ids], 1, ol, "all")[npr:]
        feats.append(R)
        labels += [int(i >= onset) for i in range(len(R))]
        groups += [r["id"]] * len(R)
    if feats:
        F = np.concatenate(feats)
        res["onset"] = {v: cv_auc(F[:, i], labels, groups=groups) for i, v in enumerate(ol)}
    (j / "probes.json").write_text(json.dumps(res, indent=1))
    f = lambda d: [d["auc_mean"], d["auc_std"], d["null_mean"], d["null_std"]] if d else ["-"] * 4
    text = ("# Probes\n\nEvidence: `results/wsp/corpus/probes.json`. StandardScaler + LogisticRegression, 5-fold CV AUC, "
            "null = 5 label shuffles.\n")
    for name, d in res["prompt_end"].items():
        text += f"\n## Prompt-end residual -> {name} (n={len(rows)})\n\n" + table(
            ["virtual layer", "AUC", "std", "null", "null std"], [[v] + f(d[v]) for v in layers])
    text += (f"\n## Repetition onset (loop rows: {len(loop_rows)}, tokens: {len(labels)}; GroupKFold by item)\n\n"
             + table(["virtual layer", "AUC", "std", "null", "null std"], [[v] + f(d) for v, d in res["onset"].items()]))
    (md / "03-probes.md").write_text(text)


# ---------- stage 5/6: ablations ----------
def pick_balanced(rows, n):
    by = {s: [r for r in rows if r["source"] == s] for s in ("gsm8k", "mmlu_pro")}
    out = []
    for i in range(n):
        for s in by:
            if i < len(by[s]) and len(out) < n:
                out.append(by[s][i])
    out += [r for r in rows if r not in out][:n - len(out)]
    return out[:n]


def mean_effect(model, group, comp, loops=1):
    from dory.wsp.lens import ablate
    return float(np.mean([ablate(model, ids, npr, loops, comp, "zero") for ids, npr in group]))


def measure(model, groups, comps, loops=1, res=None):
    res = res if res is not None else {}
    for c in comps:
        t, f = mean_effect(model, groups["trunc"], c, loops), mean_effect(model, groups["fin"], c, loops)
        res[c] = {"trunc": t, "fin": f, "diff": t - f}
    return res


def run_ablations(model, groups, loops=1, top_attn=5, top_mlp=5, n_neurons=64):
    """groups: {"trunc": [(ids, n_prompt)], "fin": [...]}. Returns {component: {trunc, fin, diff}}."""
    cfg = model.model.config
    n_phys = 2 * len(model.model.model.layers)
    res = measure(model, groups, [f"layer.{p}" for p in range(n_phys)], loops)
    top = lambda par, k: sorted((p for p in range(par, n_phys, 2)),
                                key=lambda p: -abs(res[f"layer.{p}"]["diff"]))[:k]
    attn, mlps = top(0, top_attn), top(1, top_mlp)
    measure(model, groups, [f"head.{p}.{h}" for p in attn for h in range(cfg.num_attention_heads)], loops, res)
    import torch
    mean_act = {}
    for g in ("trunc", "fin"):
        keys = {f"mlp_act.{p}" for p in mlps}
        sums = {p: 0 for p in mlps}
        for ids, npr in groups[g]:
            _, acts = model.forward(torch.as_tensor(ids).reshape(1, -1), loops, keys)
            for p in mlps:
                sums[p] = sums[p] + acts[f"mlp_act.{p}"][npr:].mean(0).cpu().numpy()
        mean_act[g] = {p: sums[p] / len(groups[g]) for p in mlps}
    comps = []
    for p in mlps:
        idx = np.argsort(-np.abs(mean_act["trunc"][p] - mean_act["fin"][p]))[:n_neurons]
        comps.append(f"neurons.{p}:" + ",".join(str(int(i)) for i in idx))
    measure(model, groups, comps, loops, res)
    return res


def build_groups(rows, tax, items, tok, n):
    by_id = {it.id: it for it in items}
    cls = {x["id"]: x["class"] for x in tax}
    sel = {"trunc": pick_balanced([r for r in rows if cls.get(r["id"]) in ("loop", "overlong")], n),
           "fin": pick_balanced([r for r in rows if not is_trunc(r) and not r.get("error") and r["correct"]], n)}
    return sel, {g: [row_ids(tok, r, by_id[r["id"]]) for r in rs] for g, rs in sel.items()}


def ablation_table(res, k=20):
    top = sorted(res.items(), key=lambda kv: -abs(kv[1]["diff"]))[:k]
    return table(["component", "d logprob truncated", "d logprob finished", "differential"],
                 [[c if len(c) < 40 else c[:30] + "...", v["trunc"], v["fin"], v["diff"]] for c, v in top])


def stage_ablations(rows, items, out_root: Path, model, tok, n=20):
    j, md = dirs(out_root)
    if (j / "ablations.json").exists():
        return print("skip ablations")
    tax = json.loads((j / "taxonomy.json").read_text())
    sel, groups = build_groups(rows, tax, items, tok, n)
    res = run_ablations(model, groups)
    (j / "ablations.json").write_text(json.dumps(
        {"items": {g: [r["id"] for r in rs] for g, rs in sel.items()}, "components": res}, indent=1))
    (md / "04-ablations.md").write_text(
        "# Component ablations (teacher-forced logprob of own continuation, loops=1, mode zero)\n\n"
        f"Evidence: `results/wsp/corpus/ablations.json`. {n} truncated (loop/overlong) vs {n} finished-correct rows; "
        "effect = mean change in logprob; differential = truncated minus finished. Note: head/neuron \"zero\" is "
        "renormalized by nGPT's L2 norm and is not comparable to layer skip (Task 6 review).\n\n"
        f"## Top 20 by |differential|\n\n{ablation_table(res)}\n")


def stage_fp32check(rows, items, out_root: Path, tok, components=None):
    import torch
    from dory.wsp.model import DoryTorch
    j, md = dirs(out_root)
    if (j / "fp32check.json").exists():
        return print("skip fp32check")
    ab = json.loads((j / "ablations.json").read_text())
    comps = components or [c for c, _ in sorted(ab["components"].items(), key=lambda kv: -abs(kv[1]["diff"]))[:5]]
    by_id = {r["id"]: r for r in rows}
    items_by = {it.id: it for it in items}
    groups = {g: [row_ids(tok, by_id[i], items_by[i]) for i in ids] for g, ids in ab["items"].items()}
    model = DoryTorch.load(BF16, dtype=torch.float32)
    res = measure(model, groups, comps)
    (j / "fp32check.json").write_text(json.dumps(res, indent=1))
    body = [[c if len(c) < 40 else c[:30] + "...", ab["components"][c]["trunc"], res[c]["trunc"],
             ab["components"][c]["fin"], res[c]["fin"], ab["components"][c]["diff"], res[c]["diff"]] for c in comps]
    with open(md / "04-ablations.md", "a") as f:
        f.write("\n## bf16 vs fp32 re-measurement\n\n" + table(
            ["component", "trunc bf16", "trunc fp32", "fin bf16", "fin fp32", "diff bf16", "diff fp32"], body))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["baselines", "taxonomy", "lens", "probes", "ablations", "fp32check"])
    ap.add_argument("--n", type=int)
    ap.add_argument("--components", nargs="*")
    ap.add_argument("--out-root", type=Path, default=ROOT)
    a = ap.parse_args()
    if a.stage == "baselines":
        return stage_baselines(a.out_root)
    rows = load_research_rows()
    if a.stage == "taxonomy":
        return stage_taxonomy(rows, a.out_root)
    items, tok = evalsets.load_split("train"), get_tok()
    if a.stage == "fp32check":
        return stage_fp32check(rows, items, a.out_root, tok, a.components)
    from dory.wsp.model import DoryTorch
    model = DoryTorch.load(BF16)
    if a.stage == "lens":
        stage_lens(rows, items, a.out_root, model, tok, a.n or 30)
    elif a.stage == "probes":
        stage_probes(rows, items, a.out_root, model, tok)
    else:
        stage_ablations(rows, items, a.out_root, model, tok, a.n or 20)


if __name__ == "__main__":
    main()
