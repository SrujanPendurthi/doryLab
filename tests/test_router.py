from pathlib import Path

import numpy as np

import dory.router as R


class FakeServer:
    def __init__(self, *a, **k):
        self.url = "http://x"

    def stop(self):
        pass


class FavourK1:
    classes_ = np.array([1, 2, 4])

    def predict_proba(self, X):
        return np.tile([0.9, 0.05, 0.05], (len(X), 1))


def _router(monkeypatch, is_code, policy):
    monkeypatch.setattr(R, "Server", FakeServer)
    monkeypatch.setattr(R, "text_features",
                        lambda p: (np.zeros((len(p), 9)), np.array([is_code]), np.zeros(len(p))))
    monkeypatch.setattr(R, "load_policy", lambda p: policy)
    return R.Router(Path("p.joblib") if policy else None, Path("m.gguf"), "f16")


def test_decide_code_floor(monkeypatch):
    pol = {"kind": "lr", "model": FavourK1(), "tau": 0.5, "feature_set": "text"}
    r = _router(monkeypatch, True, pol)
    k, secs = r.decide("def f(): pass")
    assert k >= 2 and secs >= 0
    assert _router(monkeypatch, False, pol).decide("hi")[0] == 1


def test_no_policy_defaults_k2(monkeypatch):
    assert _router(monkeypatch, False, None).decide("2+2?")[0] == 2
