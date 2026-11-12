"""Per-request loop-count router: pick k from the prompt, send it to the llama-server running that k."""
import argparse
import time
from pathlib import Path

import numpy as np

from dory.client import Completion, complete
from dory.config import KS
from dory.features import text_features
from dory.policy import choose_k, load_policy, proba_for_ks
from dory.server import Server, free_port, server_args

LOOKUP_ARGS: list[str] = []  # filled from Task 3 results once n-gram lookup support is confirmed


class Router:
    def __init__(self, policy_path: Path | None, gguf: Path, kv_type: str, ks=KS, slots: int = 4,
                 lookup_for_code: bool = False):
        if lookup_for_code and not LOOKUP_ARGS:
            raise NotImplementedError("lookup_for_code needs LOOKUP_ARGS from Task 3")
        self.policy = load_policy(policy_path) if policy_path else None
        self.feature_set = self.policy["feature_set"] if self.policy else "text"
        self.ks = list(ks) if self.policy else [2]  # no policy: fixed k=2
        self._servers: list[Server] = []
        self.servers: dict[int, Server] = {}
        self.probe_url = None
        try:
            for k in self.ks:
                self.servers[k] = self._start(server_args(gguf, k, free_port(), slots=slots, kv_type=kv_type))
            if self.feature_set != "text":
                port = free_port()
                self._start(server_args(gguf, 1, port, slots=1, ctx_per_slot=8192, embeddings=True), port)
                self.probe_url = self._servers[-1].url
        except BaseException:
            self.close()
            raise

    def _start(self, args, port=None):
        s = Server(args, port or int(args[args.index("--port") + 1]))
        self._servers.append(s)
        return s

    def decide(self, prompt: str) -> tuple[int, float]:
        t0 = time.perf_counter()
        X, is_code, _ = text_features([prompt])
        if self.policy is None:
            k = int(choose_k(np.array([[0, 1, 0]]), is_code, 1.0)[0])  # fixed k=2, code floor is moot
        else:
            if self.feature_set != "text":
                from dory.probe import probe_features  # lazy: only probe policies need it
                P = probe_features([prompt], self.probe_url)
                X = P if self.feature_set == "probe" else np.hstack([P, X])
            k = int(choose_k(proba_for_ks(self.policy["model"], X), is_code, self.policy["tau"],
                             ks=self.ks)[0])
        return k, time.perf_counter() - t0

    def generate(self, prompt: str, max_tokens: int = 512) -> tuple[Completion, int]:
        k, _ = self.decide(prompt)
        return complete(self.servers[k].url, prompt, max_tokens), k

    def close(self):
        for s in self._servers:
            try:
                s.stop()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path, default=None)
    ap.add_argument("--gguf", type=Path, default=Path("models/dory-q5.gguf"))
    ap.add_argument("--kv", default="f16")
    ap.add_argument("prompt")
    a = ap.parse_args()
    with Router(a.policy, a.gguf, a.kv) as r:
        k, secs = r.decide(a.prompt)
        c = complete(r.servers[k].url, a.prompt, 512)
        print(f"k={k} controller={secs * 1000:.1f} ms gen={c.gen_tps:.1f} tok/s\n{c.text}")


if __name__ == "__main__":
    main()
