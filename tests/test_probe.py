import numpy as np

from dory import probe


def test_probe_features_normalized(monkeypatch):
    monkeypatch.setattr(probe, "embed_last", lambda url, text: [3.0, 4.0])
    X = probe.probe_features(["a", "b"], "http://x")
    assert X.shape == (2, 2)
    assert np.allclose(np.linalg.norm(X, axis=1), 1.0)


def test_probe_seconds():
    labels = [{"id": "a", "tokens_by_k": {"1": [1000, 5], "2": [1000, 7]}},
              {"id": "b", "tokens_by_k": {2: [500, 7]}}]
    out = probe.probe_seconds(labels, {1: {"prompt_tps": 500}})
    assert out == {"a": 2.0, "b": 1.0}
