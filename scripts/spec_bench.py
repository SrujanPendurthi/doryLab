"""Task 11 Step 3: self-speculative k=4 (k=1 draft) on 100 holdout prompts, slots=1."""
import json
from pathlib import Path
from dory.bench import load_speed
from dory.client import complete
from dory.evalsets import MAX_TOKENS, load_split
from dory.server import Server, free_port, server_args

STOCK = Path("models/gguf/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf")  # metadata num_loops=4
DRAFT = Path("models/gguf/dory-q5-loops1-draft.gguf")
items = load_split("holdout")
pick = [it for s in ("gsm8k", "mmlu_pro", "mbpp") for it in [i for i in items if i.source == s][:34]][:100]
N_ID = 20  # identity check subset

def run(args, its):
    port = free_port()
    with Server(args(port), port, startup_timeout=300) as s:
        return [complete(s.url, it.prompt, MAX_TOKENS[it.source]) for it in its]

spec = run(lambda p: server_args(STOCK, 4, p, slots=1, draft=DRAFT), pick)
plain = run(lambda p: server_args(STOCK, 4, p, slots=1), pick[:N_ID])  # override=4: same as stock metadata
dn, da = sum(c.draft_n for c in spec), sum(c.draft_accepted for c in spec)
gen = sum(c.n_gen for c in spec) / sum(c.n_gen / c.gen_tps for c in spec)
plain_tps = sum(c.n_gen for c in plain) / sum(c.n_gen / c.gen_tps for c in plain)
same = [a.text == b.text for a, b in zip(spec, plain)]
res = {"n": len(spec), "acceptance": da / max(dn, 1), "draft_n": dn, "spec_gen_tps": gen,
       "plain_gen_tps_measured": plain_tps, "speed_json_k4_gen_tps": load_speed(Path("results/speed.json"))[4]["gen_tps"],
       "identical": sum(same), "n_identity": len(same),
       "mismatch_ids": [it.id for it, ok in zip(pick, same) if not ok]}
res["speedup"] = gen / res["speed_json_k4_gen_tps"]
res["gate"] = res["acceptance"] >= 0.80 and res["speedup"] >= 1.2 and all(same)
print(json.dumps(res, indent=1))
Path("results/spec_k4.json").write_text(json.dumps(res, indent=1))
