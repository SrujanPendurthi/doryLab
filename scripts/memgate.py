"""Step 5 gate: 4 servers (k=1,2,4 + k=1 embeddings) up at once; system-wide memory delta + swap growth."""
import re
import subprocess
import sys
from pathlib import Path

from dory.client import complete, embed_last
from dory.bench import GGUF, bench_prompts
from dory.server import Server, free_port, server_args


def used_gb():
    out = subprocess.check_output(["vm_stat"], text=True)
    ps = int(re.search(r"page size of (\d+)", out).group(1))
    g = lambda n: int(re.search(rf"{n}:\s+(\d+)", out).group(1))
    return (g("Pages active") + g("Pages wired down") + g("Pages occupied by compressor")) * ps / 1e9


def swap_mb():
    return float(re.search(r"used = ([\d.]+)M", subprocess.check_output(["sysctl", "vm.swapusage"], text=True)).group(1))


def pressure():
    return subprocess.check_output("memory_pressure | tail -1", shell=True, text=True).strip()


SLOTS = int(sys.argv[1]) if len(sys.argv) > 1 else 1  # router default: 4 (embeddings server stays 1x8192 as in Router)
p = bench_prompts()[0]
print("before:", pressure(), f"used={used_gb():.2f}GB swap={swap_mb()}MB", flush=True)
u0, s0 = used_gb(), swap_mb()
srvs = []
try:
    for k, emb in ((1, False), (2, False), (4, False), (1, True)):
        port = free_port()
        srvs.append((k, emb, Server(server_args(GGUF, k, port, slots=1 if emb else SLOTS, ctx_per_slot=8192 if emb else 4096, embeddings=emb), port, startup_timeout=300)))
    for k, emb, s in srvs:
        embed_last(s.url, p) if emb else complete(s.url, p, 8)
    u1, s1 = used_gb(), swap_mb()
    print("after:", pressure(), f"used={u1:.2f}GB swap={s1}MB")
    print(f"DELTA used={u1 - u0:.2f}GB swap_growth={s1 - s0:.0f}MB")
    print("RSS MB:", [round(s.rss_mb()) for _, _, s in srvs], "sum", round(sum(s.rss_mb() for _, _, s in srvs)))
    for k, emb, s in srvs:
        print(k, emb, subprocess.run(["footprint", "-p", str(s.pid)], capture_output=True, text=True).stdout[-300:])
finally:
    for _, _, s in srvs:
        s.stop()
