import os
import socket
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

# repo root = parent of the `dory` package dir (works from a worktree too)
def _bin() -> Path:
    return Path(os.environ.get("DORY_LLAMA_BIN") or Path(__file__).resolve().parent.parent / "vendor/llama.cpp/build/bin")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def server_args(gguf: Path, k: int, port: int, slots: int = 4, ctx_per_slot: int = 4096,
                kv_type: str = "f16", embeddings: bool = False, draft: Path | None = None,
                extra: list[str] = []) -> list[str]:
    a = ["-m", str(gguf), "-ngl", "99", "-fa", "on", "--temp", "0", "--seed", "0", "-np", str(slots),
         "-c", str(ctx_per_slot * slots), "--port", str(port)]
    if draft is None:
        a += ["--override-kv", f"dory.num_loops=int:{k}"]
    else:
        # llama-server copies the target's kv_overrides to the draft (common/speculative.cpp
        # `result = params`), so loops come from GGUF metadata: target = stock GGUF (num_loops=4),
        # draft = APFS clone with num_loops=1 (scripts/make_draft.sh).
        if k != 4:
            raise ValueError("self-speculative draft only supported for k=4 (stock GGUF metadata)")
        a += ["-md", str(draft), "-ngld", "99", "--spec-type", "draft-simple", "--spec-draft-n-max", "4"]
    if kv_type == "q8_0":
        a += ["-ctk", "q8_0", "-ctv", "q8_0"]
    if embeddings:
        a += ["--embeddings", "--pooling", "last"]
    return a + list(extra)


class Server:
    def __init__(self, args: list[str], port: int, startup_timeout: float = 120):
        self.url = f"http://127.0.0.1:{port}"
        self._err = tempfile.TemporaryFile()
        try:
            self._p = subprocess.Popen([str(_bin() / "llama-server"), *args], stdout=subprocess.DEVNULL, stderr=self._err)
        except OSError as e:  # missing binary
            self._err.close()
            raise RuntimeError(f"cannot start llama-server: {e}") from e
        self.pid = self._p.pid
        try:
            end = time.time() + startup_timeout
            while True:
                if self._p.poll() is not None:
                    self._fail(f"llama-server exited with {self._p.returncode}")
                if self._healthy():
                    return
                if time.time() > end:
                    self._fail(f"llama-server not ready after {startup_timeout}s")
                time.sleep(0.25)
        except BaseException:  # incl. KeyboardInterrupt: never orphan the child
            self.stop()
            raise

    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=2) as r:
                return r.status == 200
        except OSError:  # incl. 503 while loading
            return False

    def _fail(self, msg: str):
        tail = self._tail()
        self.stop()
        raise RuntimeError(f"{msg}; stderr tail:\n{tail}")

    def _tail(self) -> str:
        self._err.seek(0)
        return self._err.read().decode(errors="replace")[-2000:]

    def rss_mb(self) -> float:
        return int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(self.pid)])) / 1024

    def stop(self):
        if self._p.poll() is None:
            self._p.terminate()
            try:
                self._p.wait(10)
            except subprocess.TimeoutExpired:
                self._p.kill()
                self._p.wait()
        self._err.close()  # idempotent

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()
