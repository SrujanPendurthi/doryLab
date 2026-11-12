"""Loop-count policy: classifier -> P(k) -> threshold on cumulative probability."""
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dory.config import KS, MIN_CODE_K

FEATURE_SETS = {"text", "probe", "probe+text"}


def fit(X, y, cv: bool = False):
    clf = (LogisticRegressionCV(Cs=10, cv=5, max_iter=2000, class_weight="balanced") if cv
           else LogisticRegression(max_iter=2000, class_weight="balanced"))
    return make_pipeline(StandardScaler(), clf).fit(X, y)


def proba_for_ks(model, X) -> np.ndarray:
    """predict_proba placed into KS column order; zeros for classes absent from training."""
    p = model.predict_proba(X)
    out = np.zeros((len(p), len(KS)))
    for j, c in enumerate(model.classes_):
        out[:, KS.index(int(c))] = p[:, j]
    return out


def choose_k(proba, is_code, tau: float, ks=KS, min_code_k=MIN_CODE_K) -> np.ndarray:
    """Smallest k whose cumulative P(label <= k) >= tau; max k if none reaches it (float error)."""
    cum = np.cumsum(np.asarray(proba), axis=1)
    idx = np.where((cum >= tau).any(1), (cum >= tau).argmax(1), len(ks) - 1)
    k = np.asarray(ks)[idx]
    return np.where(np.asarray(is_code, bool), np.maximum(k, min_code_k), k)


def tau_for_mean_k(proba, is_code, target: float) -> float:
    """Bisection on tau (mean k is non-decreasing in tau); uses predictions only."""
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if choose_k(proba, is_code, mid).mean() < target:
            lo = mid
        else:
            hi = mid
    return hi


def save_policy(path, kind: str, model, tau: float, feature_set: str):
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {sorted(FEATURE_SETS)}")
    joblib.dump({"kind": kind, "model": model, "tau": tau, "feature_set": feature_set}, path)


def load_policy(path) -> dict:
    return joblib.load(path)


def main(argv=None):
    import argparse
    import json
    from pathlib import Path

    from dory.bench import load_speed
    from dory.evaluate import _correct, fixed, oracle, paired_bootstrap, policy_metrics, random_matched, report
    from dory.features import text_features
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["text"], required=True)
    ap.add_argument("--results", default="results")
    R = Path(ap.parse_args(argv).results)
    speed = load_speed(R / "speed.json")
    load = lambda s: [json.loads(l) for l in (R / f"labels_{s}.jsonl").read_text().splitlines() if l.strip()]
    train, hold = load("train"), load("holdout")
    # train X from cache if present; holdout always computed (need measured seconds)
    tp = R / "features_text_train.npy"
    if tp.exists() and len(Xtr := np.load(tp)) == len(train):
        pass
    else:
        Xtr = text_features([r["prompt"] for r in train])[0]
        np.save(tp, Xtr)
    Xho, code_ho, secs = text_features([r["prompt"] for r in hold])
    np.save(R / "features_text_holdout.npy", Xho)  # X only: probe.py hstacks it
    keep = np.array([r["label_k"] is not None for r in train])
    model = fit(Xtr[keep], np.array([r["label_k"] for r in train])[keep], cv=True)
    proba = proba_for_ks(model, Xho)
    tau = tau_for_mean_k(proba, code_ho, 2.0)
    save_policy(R / "policy_text.joblib", "text", model, tau, feature_set="text")
    choice = {r["id"]: int(k) for r, k in zip(hold, choose_k(proba, code_ho, tau))}
    extra = {r["id"]: float(s) for r, s in zip(hold, secs)}
    pols = {"text": (choice, extra)}
    pols.update({f"fixed{k}": (fixed(hold, k), None) for k in KS})
    pols["random_matched"] = (random_matched(hold, 2.0), None)
    pols["oracle"] = (oracle(hold), None)
    out = report(hold, pols, speed)
    m_t, m_2 = (policy_metrics(hold, c, speed, e) for c, e in (pols["text"], pols["fixed2"]))
    d, lo, hi = paired_bootstrap([_correct(r, choice[r["id"]]) for r in hold],
                                 [_correct(r, 2) for r in hold])
    ok = d >= 0.01 or (lo <= 0 <= hi and m_t["mean_seconds"] <= 0.85 * m_2["mean_seconds"])
    out += f"\n\ntau={tau:.4f} gate({'PASS' if ok else 'FAIL'}): dacc={d:+.3f} [{lo:+.3f}, {hi:+.3f}], " \
           f"seconds {m_t['mean_seconds']:.2f} vs fixed2 {m_2['mean_seconds']:.2f}"
    print(out)
    with open(R / "decisions.md", "a") as f:
        f.write(f"\n## Task 8\n\n{out}\n")


if __name__ == "__main__":
    main()
