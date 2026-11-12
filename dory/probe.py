"""Hidden-state probe: k=1 last-token embedding -> loop-count classifier."""
import argparse
import json
from pathlib import Path

import numpy as np

from dory.client import embed_last
from dory.config import KS
from dory.evaluate import _at, fixed, oracle, report
from dory.policy import choose_k, fit, proba_for_ks, tau_for_mean_k

# ponytail: final-layer last-token embedding; mid-layer via eval-callback hook if probe underperforms


def probe_features(prompts: list[str], url: str) -> np.ndarray:
    X = np.array([embed_last(url, p) for p in prompts], float)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def probe_seconds(labels, speed) -> dict[str, float]:
    """Modeled probe tax: n_prompt / speed[1].prompt_tps (n_prompt is k-independent)."""
    return {r["id"]: next(iter(r["tokens_by_k"].values()))[0] / _at(speed, 1)["prompt_tps"]
            for r in labels}


def _load(split):
    return [json.loads(l) for l in Path(f"results/labels_{split}.jsonl").read_text().splitlines() if l.strip()]


def _cached_probe(split, prompts, url):
    path = Path(f"results/features_probe_{split}.npy")
    if path.exists() and len(X := np.load(path)) == len(prompts):
        return X
    X = probe_features(prompts, url)
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
    port = free_port()
    with Server(server_args(Path("models/dory-q5.gguf"), 1, port, slots=1, kv_type=kv,
                            embeddings=True), port) as srv:
        P = {s: _cached_probe(s, prompts[s], srv.url) for s in prompts}
    T = {s: text_features(prompts[s]) for s in prompts}
    keep = np.array([r["label_k"] is not None for r in train])
    y = np.array([r["label_k"] for r in train])[keep]
    tax = probe_seconds(hold, speed)
    pols = {f"fixed{k}": (fixed(hold, k), None) for k in KS}
    pols["oracle"] = (oracle(hold), None)
    for name, tr, ho, extra in [
            ("probe", P["train"], P["holdout"], tax),
            ("probe+text", np.hstack([P["train"], T["train"][0]]), np.hstack([P["holdout"], T["holdout"][0]]),
             {r["id"]: tax[r["id"]] + s for r, s in zip(hold, T["holdout"][2])})]:
        model = fit(tr[keep], y, cv=True)
        proba = proba_for_ks(model, ho)
        tau = tau_for_mean_k(proba, T["holdout"][1], 2.0)
        ks = choose_k(proba, T["holdout"][1], tau)
        pols[name] = ({r["id"]: int(k) for r, k in zip(hold, ks)}, extra)
    out = report(hold, pols, speed)
    print(out)
    with open("results/decisions.md", "a") as f:
        f.write(f"\n## Probe holdout report (kv={kv})\n\n{out}\n")


if __name__ == "__main__":
    main()
