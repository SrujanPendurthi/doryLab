"""Offline policy evaluation with modeled (never wall-clock) latency."""
import numpy as np

from dory.config import KS


def modeled_seconds(n_prompt, n_gen, k, speed):
    return n_prompt / speed[k]["prompt_tps"] + n_gen / speed[k]["gen_tps"]


def _at(d, k):
    """d[k] tolerating str keys (JSON round-trip)."""
    return d[k] if k in d else d[str(k)]


def _correct(rec, k):
    return bool(_at(rec["correct_by_k"], k))


def policy_metrics(labels, choice, speed, extra_seconds=None):
    extra = extra_seconds or {}
    ks = [choice[r["id"]] for r in labels]
    secs = [modeled_seconds(*_at(r["tokens_by_k"], k), k, speed) + extra.get(r["id"], 0.0)
            for r, k in zip(labels, ks)]
    acc = [_correct(r, k) for r, k in zip(labels, ks)]
    return {"acc": float(np.mean(acc)), "mean_k": float(np.mean(ks)),
            "mean_seconds": float(np.mean(secs))}


def fixed(labels, k):
    return {r["id"]: k for r in labels}


def oracle(labels):
    return {r["id"]: (r["label_k"] if r.get("label_k") is not None else 1) for r in labels}


def random_matched(labels, mean_k, seed=0):
    ks = sorted(KS)
    if mean_k <= ks[0]:
        lo = hi = ks[0]
    elif mean_k >= ks[-1]:
        lo = hi = ks[-1]
    else:
        lo = max(k for k in ks if k <= mean_k)
        hi = min(k for k in ks if k >= mean_k)
    if lo == hi:
        return fixed(labels, lo)
    rng = np.random.default_rng(seed)
    p = (mean_k - lo) / (hi - lo)
    return {r["id"]: (hi if rng.random() < p else lo) for r in labels}


def paired_bootstrap(a, b, n=10000, seed=0):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def report(labels, policies, speed):
    pols = dict(policies)
    pols.setdefault("fixed2", (fixed(labels, 2), None))
    base = [_correct(r, pols["fixed2"][0][r["id"]]) for r in labels]
    rows = ["policy | acc | mean_k | mean_seconds | Δacc vs fixed2 [95% CI]",
            "--- | --- | --- | --- | ---"]
    for name, (choice, extra) in policies.items():
        m = policy_metrics(labels, choice, speed, extra)
        d, lo, hi = paired_bootstrap([_correct(r, choice[r["id"]]) for r in labels], base)
        rows.append(f"{name} | {m['acc']:.3f} | {m['mean_k']:.2f} | {m['mean_seconds']:.2f} | "
                    f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    return "\n".join(rows)
