"""Draft-length sweep for self-speculative k=4: per-prompt acceptance + tok/s per config.
Answers: best fixed n_max, and headroom of a per-prompt n_max oracle (needed before a confidence-driven draft length)."""
import json
from pathlib import Path
from dory.client import complete
from dory.evalsets import load_split
from dory.server import Server, free_port, server_args

STOCK = Path("models/gguf/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf")
DRAFT = Path("models/gguf/dory-q5-loops1-draft.gguf")
DRAFT2 = Path("models/gguf/dory-q5-loops2-draft.gguf")  # same clone trick, num_loops=2
MAXTOK = 256  # prefix is enough to measure acceptance/speed
items = load_split("holdout")
pick = [it for s in ("gsm8k", "mmlu_pro", "mbpp") for it in [i for i in items if i.source == s][40:48]]
CONFIGS = [("plain", None), ("n2", ["--spec-draft-n-max", "2"]), ("n4", ["--spec-draft-n-max", "4"]),
           ("n6", ["--spec-draft-n-max", "6"]), ("n8", ["--spec-draft-n-max", "8"]),
           ("n8_p07", ["--spec-draft-n-max", "8", "--spec-draft-p-min", "0.7"]),
           ("d2_n4", ["--spec-draft-n-max", "4"]), ("d2_n8", ["--spec-draft-n-max", "8"]),
           # CPU/GPU split tests (6 prompts each): draft on CPU; target partly on CPU (72 layers: 16+40 looped+16)
           ("n4_cpudraft", ["--spec-draft-n-max", "4", "-ngld", "0"]),
           ("plain_ngl64", ["-ngl", "64"]), ("plain_ngl54", ["-ngl", "54"])]
SHORT = {"n4_cpudraft", "plain_ngl64", "plain_ngl54"}
out = Path("results/draft_sweep.jsonl")
done = {(r["cfg"], r["id"]) for r in map(json.loads, out.open())} if out.exists() else set()
for name, extra in CONFIGS:
    todo = [it for it in (pick[::4] if name in SHORT else pick) if (name, it.id) not in done]
    if not todo:
        continue
    port = free_port()
    spec = extra is not None and not name.startswith("plain")
    args = server_args(STOCK, 4, port, slots=1, draft=(DRAFT2 if name.startswith("d2") else DRAFT) if spec else None)
    if extra:  # later flags override server_args' default n-max
        args += extra
    with Server(args, port, startup_timeout=300) as s:
        for it in todo:
            c = complete(s.url, it.prompt, MAXTOK)
            with out.open("a") as f:
                f.write(json.dumps({"cfg": name, "id": it.id, "source": it.source, "n_gen": c.n_gen,
                                    "gen_tps": c.gen_tps, "draft_n": c.draft_n, "accepted": c.draft_accepted,
                                    "text": c.text}) + "\n")
    print(name, "done", flush=True)
