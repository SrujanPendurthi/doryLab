"""Step 6: llama-server n-gram speculative (ngram-simple) vs plain, k=2, greedy, 10 MBPP prompts with code context."""
import json
import sys
import time

from dory import evalsets
from dory.bench import GGUF
from dory.client import complete
from dory.server import Server, free_port, server_args

ds = evalsets._ds("mbpp", "test")
prompts = [f"Here is a reference implementation:\n```python\n{ds[i]['code']}\n```\n\nRewrite it with type hints and a docstring. "
           "Return only the function in a ```python code block." for i in range(10)]
res = {}
for name, extra in (("plain", []), ("ngram-simple", ["--spec-type", "ngram-simple"])):
    port = free_port()
    with Server(server_args(GGUF, 2, port, slots=1, extra=extra), port, startup_timeout=300) as s:
        complete(s.url, prompts[0][:300], 8)
        rows = [complete(s.url, p, 300) for p in prompts]
    res[name] = dict(gen_tps=sum(r.gen_tps for r in rows) / 10, tokens=sum(r.n_gen for r in rows),
                     secs=sum(r.seconds for r in rows), draft_n=sum(r.draft_n for r in rows),
                     draft_acc=sum(r.draft_accepted for r in rows), texts=[r.text for r in rows])
    print(name, {k: v for k, v in res[name].items() if k != "texts"}, flush=True)
print("speedup (mean gen_tps):", res["ngram-simple"]["gen_tps"] / res["plain"]["gen_tps"])
print("same outputs:", sum(a == b for a, b in zip(res["plain"]["texts"], res["ngram-simple"]["texts"])), "/10")
