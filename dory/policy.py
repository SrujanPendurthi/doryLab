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
