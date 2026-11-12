import subprocess
from pathlib import Path

import pytest


from dory.server import Server, free_port, server_args


def test_server_args_sets_loops():
    assert "dory.num_loops=int:4" in server_args(Path("m.gguf"), 4, 8080)


def test_server_args_q8_kv():
    a = server_args(Path("m.gguf"), 1, 8080, kv_type="q8_0")
    i = a.index("-ctk")
    assert a[i : i + 4] == ["-ctk", "q8_0", "-ctv", "q8_0"]
    assert "-ctk" not in server_args(Path("m.gguf"), 1, 8080)


def test_server_args_embeddings_and_ctx():
    a = server_args(Path("m.gguf"), 1, 8080, slots=2, ctx_per_slot=100, embeddings=True)
    assert "--embeddings" in a and a[a.index("--pooling") + 1] == "last"
    assert a[a.index("-c") + 1] == "200" and a[a.index("-np") + 1] == "2"


def test_server_bad_model_raises():
    p = free_port()
    with pytest.raises(RuntimeError):
        Server(server_args(Path("/nonexistent.gguf"), 1, p), p, startup_timeout=30)
    assert subprocess.run(["pgrep", "-f", "nonexistent.gguf"], capture_output=True).returncode != 0


@pytest.mark.integration
def test_real_server_completes(gguf):
    from dory.client import complete

    p = free_port()
    with Server(server_args(gguf, 1, p), p) as s:
        assert complete(s.url, "2+2=", 8).n_gen > 0


def test_stop_idempotent_closes_stderr(monkeypatch, tmp_path):
    monkeypatch.setenv("DORY_LLAMA_BIN", str(tmp_path))
    (tmp_path / "llama-server").write_text("#!/bin/sh\nexec sleep 60\n")
    (tmp_path / "llama-server").chmod(0o755)
    monkeypatch.setattr(Server, "_healthy", lambda self: True)
    s = Server([], 1)
    s.stop()
    s.stop()
    assert s._err.closed and s._p.poll() is not None


def test_interrupt_during_startup_stops_child(monkeypatch, tmp_path):
    monkeypatch.setenv("DORY_LLAMA_BIN", str(tmp_path))
    (tmp_path / "llama-server").write_text("#!/bin/sh\nexec sleep 60\n")
    (tmp_path / "llama-server").chmod(0o755)
    made = []
    orig = Server.stop
    monkeypatch.setattr(Server, "stop", lambda self: (made.append(self), orig(self)))

    def boom(self):
        raise KeyboardInterrupt

    monkeypatch.setattr(Server, "_healthy", boom)
    with pytest.raises(KeyboardInterrupt):
        Server([], 1)
    assert made and made[0]._p.poll() is not None
