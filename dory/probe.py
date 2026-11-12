"""Hidden-state probe: k=1 last-token embedding -> loop-count classifier."""
import argparse
import json
from pathlib import Path

import numpy as np

from dory.client import embed_last
from dory.config import KS
from dory.evaluate import _at, fixed, oracle, report
from dory.policy import choose_k, fit, load_policy, proba_for_ks, tau_for_mean_k

# ponytail: final-layer last-token embedding; mid-layer via eval-callback hook if probe underperforms


def probe_features(prompts: list[str], url: str) -> np.ndarray:
    X = np.array([embed_last(url, p) for p in prompts], float)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def probe_seconds(labels, speed) -> dict[str, float]:
    """Modeled probe tax: n_prompt / speed[1].prompt_tps (n_prompt is k-independent)."""
    def n_prompt(t):
        return (_at(t, 1) if 1 in t or "1" in t else next(iter(t.values())))[0]
    return {r["id"]: n_prompt(r["tokens_by_k"]) / _at(speed, 1)["prompt_tps"] for r in labels}


def _load(split):
    return [json.loads(l) for l in Path(f"results/labels_{split}.jsonl").read_text().splitlines() if l.strip()]


def _cached_probe(split, prompts, kv, get_url):
    """get_url is called only on a cache miss, so a hit never starts a server."""
    path = Path(f"results/features_probe_{split}_{kv}.npy")
    if path.exists() and len(X := np.load(path)) == len(prompts):
        return X
    X = probe_features(prompts, get_url())
    np.save(path, X)
    return X


def main():
    from dory.bench import load_speed  # lazy: added by another task
    from dory.features import text_features
    from dory.server import Server, free_port, server_args
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv", choices=["f16", "q8_0"], default="f16")
    kv = ap.parse_args().kv
    speed = load_speed()
    train, hold = _load("train"), _load("holdout")
    prompts = {s: [r["prompt"] for r in rows] for s, rows in (("train", train), ("holdout", hold))}
    srv = None

    def get_url():
        nonlocal srv
        if srv is None:
            port = free_port()
            srv = Server(server_args(Path("models/dory-q5.gguf"), 1, port, slots=1, ctx_per_slot=8192,
                                     kv_type=kv, embeddings=True), port).__enter__()
        return srv.url
    try:
        P = {s: _cached_probe(s, prompts[s], kv, get_url) for s in prompts}
    finally:
        if srv:
            srv.stop()
    # train text X from cache if present; holdout always computed (need measured seconds)
    tp = Path("results/features_text_train.npy")
    Xt_tr = np.load(tp) if tp.exists() and len(np.load(tp)) == len(train) else text_features(prompts["train"])[0]
    Xt_ho, code_ho, text_secs = text_features(prompts["holdout"])
    T = {"train": Xt_tr, "holdout": Xt_ho}
    keep = np.array([r["label_k"] is not None for r in train])
    y = np.array([r["label_k"] for r in train])[keep].astype(int)
    tax = probe_seconds(hold, speed)
    pols = {f"fixed{k}": (fixed(hold, k), None) for k in KS}
    pols["oracle"] = (oracle(hold), None)
    tsecs = {r["id"]: float(t) for r, t in zip(hold, text_secs)}
    tpol = Path("results/policy_text.joblib")
    if tpol.exists():
        pol = load_policy(tpol)
        tm, ttau = pol["model"], pol["tau"]
    else:
        tm = fit(T["train"][keep], y, cv=True)
        ttau = tau_for_mean_k(proba_for_ks(tm, T["holdout"]), code_ho, 2.0)
    tks = choose_k(proba_for_ks(tm, T["holdout"]), code_ho, ttau)
    pols["text"] = ({r["id"]: int(k) for r, k in zip(hold, tks)}, tsecs)
    for name, tr, ho, extra in [
            ("probe", P["train"], P["holdout"], tax),
            ("probe+text", np.hstack([P["train"], T["train"]]), np.hstack([P["holdout"], T["holdout"]]),
             {r["id"]: tax[r["id"]] + s for r, s in zip(hold, text_secs)})]:
        model = fit(tr[keep], y, cv=True)
        proba = proba_for_ks(model, ho)
        tau = tau_for_mean_k(proba, code_ho, 2.0)
        ks = choose_k(proba, code_ho, tau)
        pols[name] = ({r["id"]: int(k) for r, k in zip(hold, ks)}, extra)
    out = report(hold, pols, speed)
    print(out)
    with open("results/decisions.md", "a") as f:
        f.write(f"\n## Probe holdout report (kv={kv})\n\n{out}\n")


if __name__ == "__main__":
    main()
