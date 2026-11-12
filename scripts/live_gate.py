"""Task 10 Step 5: live router gate on 50 holdout prompts (controller overhead + live vs offline accuracy)."""
import json, statistics, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from dory.evalsets import load_split, grade
from dory.evaluate import policy_metrics
from dory.bench import load_speed
from dory.features import text_features
from dory.policy import load_policy, proba_for_ks, choose_k
from dory.router import Router

R = Path("results")
labels = {r["id"]: r for r in map(json.loads, open(R / "labels_holdout.jsonl"))}
items = load_split("holdout")
pick = []
for src, n in (("gsm8k", 17), ("mmlu_pro", 17), ("mbpp", 16)):
    pick += [it for it in items if it.source == src][:n]

text_features(["warm up"])  # load classifier once; overhead is per-request steady state
pol = load_policy(R / "policy_text.joblib")
oh_text = []
for it in pick:
    t0 = time.perf_counter()
    X, code, _ = text_features([it.prompt])
    choose_k(proba_for_ks(pol["model"], X), code, pol["tau"])
    oh_text.append(time.perf_counter() - t0)

with Router(None, Path("models/dory-q5.gguf"), "f16", slots=4) as r:
    oh_live = [r.decide(it.prompt)[1] for it in pick]
    with ThreadPoolExecutor(4) as ex:
        outs = list(ex.map(r.generate, [it.prompt for it in pick]))
live = {it.id: grade(it, c.text) for it, (c, k) in zip(pick, outs)}
choice = {it.id: k for it, (c, k) in zip(pick, outs)}
off = policy_metrics([labels[i] for i in live], choice, load_speed(R / "speed.json"))
res = {"n": len(live), "live_acc": sum(live.values()) / len(live), "offline_acc": off["acc"],
       "ks": sorted(set(choice.values())),
       "overhead_ms_text_policy": 1000 * statistics.mean(oh_text),
       "overhead_ms_live_none": 1000 * statistics.mean(oh_live),
       "disagree": [i for i in live if live[i] != labels[i]["correct_by_k"][str(choice[i])]]}
res["gate"] = res["overhead_ms_text_policy"] < 50 and abs(res["live_acc"] - res["offline_acc"]) <= 0.05
print(json.dumps(res, indent=1))
(R / "live_gate.json").write_text(json.dumps(res, indent=1))
