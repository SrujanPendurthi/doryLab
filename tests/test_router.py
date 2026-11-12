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


def test_startup_failure_stops_started(monkeypatch):
    stopped, n = [], []

    class Flaky(FakeServer):
        def __init__(self, *a, **k):
            n.append(1)
            if len(n) == 2:
                raise RuntimeError("boom")
            self.url = "u"

        def stop(self):
            stopped.append(self)

    monkeypatch.setattr(R, "Server", Flaky)
    monkeypatch.setattr(R, "load_policy", lambda p: {"kind": "lr", "model": FavourK1(), "tau": 0.5,
                                                    "feature_set": "text"})
    import pytest
    with pytest.raises(RuntimeError):
        R.Router(Path("p"), Path("m"), "f16")
    assert len(stopped) == 1


def test_generate_routes_and_max_tokens(monkeypatch):
    pol = {"kind": "lr", "model": FavourK1(), "tau": 0.5, "feature_set": "text"}
    calls = []
    monkeypatch.setattr(R, "complete", lambda url, p, m: calls.append((url, m)) or "c")
    r = _router(monkeypatch, True, pol)
    r.servers[2].url = "http://k2"
    assert r.generate("code") == ("c", 2) and calls[-1] == ("http://k2", 768)
    r = _router(monkeypatch, False, pol)
    r.servers[1].url = "http://k1"
    r.generate("hi")
    assert calls[-1] == ("http://k1", 512)
    r.generate("hi", 5)
    assert calls[-1][1] == 5


def _counting(monkeypatch, r):
    n = []

    def tf(p):
        n.append(1)
        return np.zeros((len(p), 9)), np.array([False]), np.zeros(len(p))
    monkeypatch.setattr(R, "text_features", tf)
    monkeypatch.setattr(R, "complete", lambda url, p, mt: mt)
    return n


def test_no_policy_skips_classifier(monkeypatch):
    r = _router(monkeypatch, False, None)
    n = _counting(monkeypatch, r)
    assert r.decide("hi")[0] == 2 and not n
    assert r.generate("hi", 100) == (100, 2) and not n
    assert r.generate("hi") == (512, 2) and len(n) == 1
