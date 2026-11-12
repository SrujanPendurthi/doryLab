"""Task 10 Step 6: final test, run exactly once. Chosen policy = None (fixed k=2 + code floor)."""
import json
from pathlib import Path
from dory.evaluate import report, fixed, oracle
from dory.bench import load_speed
from dory.features import text_features
from dory.policy import load_policy, proba_for_ks, choose_k

R = Path("results")
L = [json.loads(l) for l in open(R / "labels_final.jsonl")]
X, code, secs = text_features([r["prompt"] for r in L])
pol = load_policy(R / "policy_text.joblib")
ks = choose_k(proba_for_ks(pol["model"], X), code, pol["tau"])  # same tau, no refit
pols = {
    "chosen=None (fixed2+floor)": (fixed(L, 2), None),
    "fixed1": (fixed(L, 1), None), "fixed4": (fixed(L, 4), None), "oracle": (oracle(L), None),
    "fixed1+code floor (comparison)": ({r["id"]: (2 if r["source"] == "mbpp" else 1) for r in L}, None),
    "text (failed gate, reference)": ({r["id"]: int(k) for r, k in zip(L, ks)},
                                      {r["id"]: float(s) for r, s in zip(L, secs)}),
}
out = report(L, pols, load_speed(R / "speed.json"))
print(out)
