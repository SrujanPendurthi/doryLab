"""Phase 0 speed/memory benchmark: python -m dory.bench --out results/bench.csv"""
import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dory import evalsets
from dory.client import Completion, complete
from dory.server import Server, free_port, server_args

GGUF = Path("models/dory-q5.gguf")
# (k, slots, kv_type): 9 f16 configs + k=4 q8_0 at slots 1 and 4
CONFIGS = [(k, s, "f16") for k in (1, 2, 4) for s in (1, 4, 8)] + [(4, 1, "q8_0"), (4, 4, "q8_0")]
FIELDS = ["k", "slots", "kv", "gen_tps_per_req", "agg_tps", "prompt_tps", "mean_n_prompt", "rss_mb"]
PROMPT_CHARS = 4000  # ~1k tokens; actual mean n_prompt is recorded per row


def summarize(rows: list[Completion], wall_seconds: float) -> dict:
    n = len(rows)
    return {"gen_tps_per_req": sum(r.gen_tps for r in rows) / n,
            "agg_tps": sum(r.n_gen for r in rows) / wall_seconds,
            "prompt_tps": sum(r.prompt_tps for r in rows) / n}


def load_speed(path: Path = Path("results/speed.json")) -> dict[int, dict[str, float]]:
    return {int(k): v for k, v in json.loads(Path(path).read_text()).items()}


def bench_prompts() -> list[str]:
    """8 deterministic ~1k-token prompts: 4 GSM8K-based, 4 MBPP-based (padded with sibling items)."""
    out = []
    for src, n in (("gsm8k", 4), ("mbpp", 4)):
        pool = [it.prompt for it in evalsets.load(src, "test", 200)]
        for j in range(n):
            parts, i = [], j
            while sum(map(len, parts)) < PROMPT_CHARS:
                parts.append(pool[i % len(pool)])
                i += n
            out.append("\n\n---\n\n".join(parts))
    return out


def run_config(k: int, slots: int, kv: str, prompts: list[str]) -> dict:
    port = free_port()
    with Server(server_args(GGUF, k, port, slots=slots, kv_type=kv), port, startup_timeout=300) as srv:
        complete(srv.url, prompts[0][:400], 8)  # warm-up
        t0 = time.time()
        with ThreadPoolExecutor(slots) as ex:
            rows = list(ex.map(lambda p: complete(srv.url, p, 300), prompts))
        wall = time.time() - t0
        return {"k": k, "slots": slots, "kv": kv, **summarize(rows, wall),
                "mean_n_prompt": sum(r.n_prompt for r in rows) / len(rows), "rss_mb": srv.rss_mb()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/bench.csv"))
    out = ap.parse_args().out
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {(int(r["k"]), int(r["slots"]), r["kv"]) for r in csv.DictReader(out.open())} if out.exists() else set()
    prompts = bench_prompts()
    for cfg in CONFIGS:
        if cfg in done:
            continue
        row = run_config(*cfg, prompts)
        new = not out.exists()
        with out.open("a", newline="") as f:
            w = csv.DictWriter(f, FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
        print(row, flush=True)
    rows = [r for r in csv.DictReader(out.open()) if r["slots"] == "1" and r["kv"] == "f16"]
    speed = {r["k"]: {"prompt_tps": float(r["prompt_tps"]), "gen_tps": float(r["gen_tps_per_req"])} for r in rows}
    out.with_name("speed.json").write_text(json.dumps(speed, indent=2))


if __name__ == "__main__":
    main()
