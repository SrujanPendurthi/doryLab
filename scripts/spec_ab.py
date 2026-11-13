"""Interleaved A/B: all configs' servers up at once, configs rotated per prompt, so machine-load drift hits every arm equally.
Settles whether self-spec n4/n6 beats plain (draft sweep ran arms hours apart)."""
import json
from contextlib import ExitStack
from pathlib import Path
from dory.client import complete
from dory.server import Server, free_port, server_args
from scripts.draft_sweep import STOCK, DRAFT, MAXTOK, pick  # noqa: E402  (draft_sweep body is resume-guarded: all rows done)

CFGS = {"plain": ([], False), "n4": (["--spec-draft-n-max", "4"], True),
        "n6": (["--spec-draft-n-max", "6"], True)}  # ngl54 dropped: 4 servers -> Metal "Compute error"
out = Path("results/spec_ab.jsonl")
done = {(r["cfg"], r["id"]) for r in map(json.loads, out.open())} if out.exists() else set()
with ExitStack() as st:
    urls = {}
    for name, (extra, spec) in CFGS.items():
        port = free_port()
        urls[name] = st.enter_context(Server(server_args(STOCK, 4, port, slots=1, draft=DRAFT if spec else None) + extra,
                                             port, startup_timeout=300)).url
    names = list(CFGS)
    for j, it in enumerate(pick):
        for name in names[j % len(names):] + names[:j % len(names)]:
            if (name, it.id) in done:
                continue
            c = complete(urls[name], it.prompt, MAXTOK)
            with out.open("a") as f:
                f.write(json.dumps({"cfg": name, "id": it.id, "source": it.source, "n_gen": c.n_gen, "gen_tps": c.gen_tps,
                                    "draft_n": c.draft_n, "accepted": c.draft_accepted, "text": c.text}) + "\n")
        print(j, "done", flush=True)
