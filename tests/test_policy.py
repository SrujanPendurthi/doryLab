import numpy as np
import pytest

from dory.config import KS
from dory.features import feature_rows
from dory.policy import (choose_k, decide_k, fit, load_policy, proba_for_ks, save_policy,
                         tau_for_mean_k)

P = np.array([[0.5, 0.3, 0.2]])


def test_decide_k_uses_tau_and_code_floor(monkeypatch):
    from dory import policy

    class FavourK1:
        classes_ = np.array(KS)

        def predict_proba(self, X):
            return np.tile([0.9, 0.05, 0.05], (len(X), 1))

    pol = {"model": FavourK1(), "tau": 0.5, "feature_set": "text"}
    monkeypatch.setattr(policy, "text_features",
                        lambda texts: (np.zeros((1, 9)), np.array([False]), np.zeros(1)))
    assert decide_k(pol, "easy") == 1
    pol["tau"] = 0.99
    assert decide_k(pol, "easy") == 4
    pol["tau"] = 0.5
    monkeypatch.setattr(policy, "text_features",
                        lambda texts: (np.zeros((1, 9)), np.array([True]), np.zeros(1)))
    assert decide_k(pol, "code") == 2


def test_decide_k_matches_router(monkeypatch):
    import json
    import threading
    from pathlib import Path
    from huggingface_hub import try_to_load_from_cache
    from dory.nvidia_classifier import REPO
    from dory.router import Router

    if not isinstance(try_to_load_from_cache(REPO, "model.safetensors"), str):
        pytest.skip("classifier weights are not cached")
    policy_path = Path("results/policy_text.joblib")
    prompts_path = Path("results/labels_train.jsonl")
    if not policy_path.exists() or not prompts_path.exists():
        pytest.skip("trained policy or train prompts unavailable")
    # CPU only; the parity test never starts a llama-server or uses a GPU.
    from dory import features, policy, router
    real_features = features.text_features
    cpu_features = lambda texts: real_features(texts, device="cpu")
    monkeypatch.setattr(policy, "text_features", cpu_features)
    monkeypatch.setattr(router, "text_features", cpu_features)
    r = Router.__new__(Router)
    r.policy = load_policy(policy_path)
    r.feature_set = "text"
    r._lock = threading.Lock()
    prompts = [json.loads(line)["prompt"] for line in prompts_path.read_text().splitlines()[:5]]
    for prompt in prompts:
        assert decide_k(r.policy, prompt) == r._decide(prompt)[0]


def test_choose_k_cumulative():
    assert choose_k(P, np.array([False]), 0.5).tolist() == [1]
    assert choose_k(P, np.array([False]), 0.8).tolist() == [2]


def test_code_never_k1():
    assert choose_k(P, np.array([True]), 0.5).tolist() == [2]


def test_choose_k_float_shortfall_gives_max():
    p = np.array([[0.3, 0.3, 0.3]])
    assert choose_k(p, np.array([False]), 1.0).tolist() == [4]


def test_tau_for_mean_k():
    proba = np.random.default_rng(0).dirichlet([1, 1, 1], size=500)
    is_code = np.zeros(500, bool)
    tau = tau_for_mean_k(proba, is_code, 2.0)
    assert abs(choose_k(proba, is_code, tau).mean() - 2.0) < 0.1


def _sep_data():
    rng = np.random.default_rng(0)
    y = np.repeat(KS, 60)
    X = rng.normal(size=(180, 3)) + 4 * np.searchsorted(KS, y)[:, None]
    return X, y


@pytest.mark.parametrize("cv", [False, True])
def test_fit_predicts_classes(cv):
    X, y = _sep_data()
    assert (fit(X, y, cv=cv).predict(X) == y).mean() > 0.9


def test_proba_for_ks_missing_class_and_roundtrip(tmp_path):
    X, y = _sep_data()
    m = fit(X[y != 2], y[y != 2])
    p = proba_for_ks(m, X)
    assert p.shape == (180, 3) and (p[:, 1] == 0).all()
    assert np.allclose(p.sum(1), 1)
    save_policy(tmp_path / "p.joblib", "text", m, 0.7, "text")
    d = load_policy(tmp_path / "p.joblib")
    assert d["tau"] == 0.7 and d["feature_set"] == "text" and d["kind"] == "text"
    with pytest.raises(ValueError):
        save_policy(tmp_path / "q.joblib", "text", m, 0.7, "bogus")


def test_feature_rows():
    res = {"creativity_scope": [.1, .2], "reasoning": [.3, .4], "contextual_knowledge": [.5, .6],
           "domain_knowledge": [.7, .8], "constraint_ct": [.9, 1.0], "number_of_few_shots": [0, 1],
           "prompt_complexity_score": [.25, .35],
           "task_type_1": ["Code Generation", "Open QA"]}
    X, is_code = feature_rows(res, ["abc", "a" * 99])
    assert X.shape == (2, 9) and is_code.tolist() == [True, False]
    assert X[0].tolist() == [.1, .3, .5, .7, .9, 0, .25, np.log1p(3), 1.0]
    assert X[1, 8] == 0.0


@pytest.mark.network
def test_text_features_shape_real():
    from dory.features import text_features
    X, is_code, seconds = text_features(["Write a Python script that uses a for loop.", "What is 2+2?"])
    assert X.shape == (2, 9) and is_code.shape == (2,) and seconds.shape == (2,)


def test_text_features_empty():
    from dory.features import text_features
    X, is_code, seconds = text_features([])
    assert X.shape == (0, 9) and is_code.shape == (0,) and seconds.shape == (0,)


def test_main_text(tmp_path, monkeypatch):
    import json
    from dory import features, policy
    rng = np.random.default_rng(0)

    def fake(prompts):
        X = rng.normal(size=(len(prompts), 9))
        X[:, 0] = [float(p) for p in prompts]
        X[:, -1] = 0
        return X, np.zeros(len(prompts), bool), np.full(len(prompts), 0.01)
    monkeypatch.setattr(features, "text_features", fake)
    (tmp_path / "speed.json").write_text(json.dumps(
        {str(k): {"prompt_tps": 100.0, "gen_tps": 10.0 / k} for k in (1, 2, 4)}))

    def rows(n, p):
        out = []
        for i in range(n):
            y = [1, 2, 4, None][i % 4]  # None = wrong at every k (real labels have these)
            out.append({"id": f"{p}{i}", "source": "gsm8k", "prompt": str(y or 0), "label_k": y,
                        "correct_by_k": {str(k): y is not None and k >= y for k in (1, 2, 4)},
                        "tokens_by_k": {str(k): [10, 5] for k in (1, 2, 4)}})
        return out
    for s, n in (("train", 30), ("holdout", 12)):
        (tmp_path / f"labels_{s}.jsonl").write_text("\n".join(json.dumps(r) for r in rows(n, s)))
    policy.main(["--kind", "text", "--results", str(tmp_path)])
    assert np.load(tmp_path / "features_text_train.npy").shape == (30, 9)
    assert np.load(tmp_path / "features_text_holdout.npy").shape == (12, 9)
    assert 0 < policy.load_policy(tmp_path / "policy_text.joblib")["tau"] <= 1
    assert "## Task 8" in (tmp_path / "decisions.md").read_text()
