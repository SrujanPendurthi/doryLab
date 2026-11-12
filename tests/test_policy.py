import numpy as np
import pytest

from dory.config import KS
from dory.features import feature_rows
from dory.policy import (choose_k, fit, load_policy, proba_for_ks, save_policy,
                         tau_for_mean_k)

P = np.array([[0.5, 0.3, 0.2]])


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
