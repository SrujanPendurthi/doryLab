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


def test_probe_seconds_prefers_k1_and_falls_back():
    sp = {1: {"prompt_tps": 100}}
    assert probe.probe_seconds([{"id": "a", "tokens_by_k": {"4": [900, 1], 1: [300, 1]}}], sp) == {"a": 3.0}
    assert probe.probe_seconds([{"id": "a", "tokens_by_k": {"4": [900, 1]}}], sp) == {"a": 9.0}


def test_cached_probe_reuse_starts_no_server(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    np.save("results/features_probe_train_f16.npy", np.ones((2, 3)))
    def boom():
        raise AssertionError("server started")
    assert probe._cached_probe("train", ["a", "b"], "f16", boom).shape == (2, 3)
    monkeypatch.setattr(probe, "embed_last", lambda u, t: [1.0, 0.0])
    assert probe._cached_probe("train", ["a", "b", "c"], "f16", lambda: "u").shape == (3, 2)
