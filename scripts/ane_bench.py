"""Core ML (ANE) vs PyTorch latency/parity for the NVIDIA prompt classifier. Writes results/ane_bench.json."""
import json, os, subprocess, statistics as st, sys, time, traceback
import numpy as np, psutil, torch, torch.nn as nn
import coremltools as ct
from dory.evalsets import load_split
from dory.features import feature_rows
from dory.nvidia_classifier import load_classifier

import math
import transformers.models.deberta_v2.modeling_deberta_v2 as _dm
# coremltools can't sqrt the int tensor torch.jit.trace makes here; fold to a constant (static shapes)
_dm.scaled_size_sqrt = lambda q, sf: torch.tensor(math.sqrt(int(q.size(-1)) * sf), dtype=torch.float)



def _static_relpos(q, k, bucket_size=-1, max_position=-1):  # constant [1,L,L]; avoids traced control flow
    n = int(q.size(-2)); ids = torch.arange(n)
    r = ids[:, None] - ids[None, :]
    if bucket_size > 0 and max_position > 0:
        r = _dm.make_log_bucket_position(r, bucket_size, max_position)
    return r.long().unsqueeze(0)
_dm.build_relative_position = _static_relpos
_dm.build_rpos = lambda q, k, relative_pos, pb, mrp: relative_pos  # self-attention only: q and k lengths match

BUCKETS = (128, 512)
OUT = "models/coreml"
UNITS = {"CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE, "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
         "CPU_ONLY": ct.ComputeUnit.CPU_ONLY, "ALL": ct.ComputeUnit.ALL}
rss = lambda: psutil.Process().memory_info().rss / 2**20
R = {"failures": []}


class Wrap(nn.Module):
    def __init__(s, m):
        super().__init__(); s.m = m
    def forward(s, input_ids, attention_mask):
        ids, am = input_ids.long(), attention_mask.long()
        pooled = s.m.pool(s.m.backbone(input_ids=ids, attention_mask=am).last_hidden_state, am)
        return tuple(s.m.heads[k](pooled) for k in range(len(s.m.heads)))


def convert(model, L):
    path = f"{OUT}/classifier_{L}.mlpackage"
    if os.path.exists(path):
        return path
    w = Wrap(model).eval()
    ex = (torch.ones(1, L, dtype=torch.int32), torch.ones(1, L, dtype=torch.int32))
    with torch.no_grad():
        tr = torch.jit.trace(w, ex, strict=False)
    ml = ct.convert(tr, convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                    inputs=[ct.TensorType("input_ids", (1, L), dtype=np.int32),
                            ct.TensorType("attention_mask", (1, L), dtype=np.int32)])
    ml.save(path)
    return path


def pct(xs):
    xs = sorted(xs); return {"median_ms": st.median(xs), "p90_ms": xs[int(0.9 * (len(xs) - 1))]}


def main():
    prompts = [it.prompt for it in load_split("holdout")[:60]]
    base = rss()
    model, tok, _ = load_classifier("cpu")
    model = model.float()
    R["rss_mb_pytorch_cpu_load"] = rss() - base
    enc = lambda p: tok(p, return_tensors="pt", add_special_tokens=True, max_length=512, truncation=True)
    lens = [enc(p)["input_ids"].shape[1] for p in prompts]
    R["token_lengths"] = {"min": min(lens), "median": st.median(lens), "p90": sorted(lens)[53], "max": max(lens),
                          "n_le_128": sum(l <= 128 for l in lens), "n_gt_128": sum(l > 128 for l in lens)}
    print("tokens", R["token_lengths"])

    # reference: CPU fp32 logits
    ref = []
    with torch.no_grad():
        for p in prompts:
            e = enc(p); ref.append([model.heads[k](model.pool(model.backbone(**e).last_hidden_state, e["attention_mask"]))
                                    for k in range(len(model.heads))])
    ref_logits = [torch.cat([r[k] for r in ref]) for k in range(len(model.heads))]
    ref_res = model.process_logits(ref_logits)
    Xr, cr = feature_rows(ref_res, prompts)

    lat = R.setdefault("latency", {})
    # PyTorch arms
    for dev in ("cpu", "mps"):
        m, _, _ = load_classifier(dev)
        ts = []
        for i, p in enumerate(prompts[:5] + prompts):
            t0 = time.perf_counter()
            e = enc(p).to(dev)
            with torch.no_grad():
                m(e)
            if dev == "mps": torch.mps.synchronize()
            if i >= 5: ts.append((time.perf_counter() - t0) * 1000)
        lat[f"pytorch_{dev}"] = pct(ts)
        print(dev, lat[f"pytorch_{dev}"])

    # Core ML
    paths = {}
    for L in BUCKETS:
        try:
            t0 = time.perf_counter(); paths[L] = convert(model, L)
            print("converted", L, round(time.perf_counter() - t0, 1), "s")
        except Exception:
            R["failures"].append({"stage": f"convert_{L}", "trace": traceback.format_exc()[-1500:]}); print(traceback.format_exc())
    R["converted_buckets"] = list(paths)
    if len(paths) < len(BUCKETS):
        return
    # Core ML dealloc segfaults (libcoremlpython, objc teardown): every arm/plan runs in its own subprocess
    # that ends with os._exit(0) after writing its JSON, so no teardown runs and a crash loses one arm.
    os.makedirs(TMP, exist_ok=True)
    torch.save({"ref": ref_logits, "Xr": Xr, "cr": cr, "prompts": prompts}, f"{TMP}/ref.pt")
    R["coreml"] = {}
    for name in list(UNITS) + ["PLAN"]:
        out = f"{TMP}/{name}.json"
        if os.path.exists(out): os.remove(out)
        r = subprocess.run([sys.executable, "-m", "scripts.ane_bench", "--arm", name], capture_output=True, text=True)
        if os.path.exists(out):
            d = json.load(open(out))
        else:
            d = {"error": f"subprocess rc={r.returncode}", "stderr_tail": r.stderr[-800:]}
        if name == "PLAN": R.update(d)
        else: R["coreml"][name] = d
        print(name, json.dumps(d)[:300], flush=True)
        json.dump(R, open("results/ane_bench.json", "w"), indent=1, default=str)


TMP = "models/coreml/tmp"


def arm(name):
    os.makedirs(TMP, exist_ok=True)
    out = {}
    paths = {L: f"{OUT}/classifier_{L}.mlpackage" for L in BUCKETS}
    keep = []  # hold every Core ML object forever; never dealloc
    try:
        if name == "PLAN":
            from coremltools.models.compute_plan import MLComputePlan
            plans = {}
            for L, p in paths.items():
                cm = ct.models.MLModel(p); keep.append(cm)
                plan = MLComputePlan.load_from_path(cm.get_compiled_model_path(), compute_units=ct.ComputeUnit.CPU_AND_NE)
                keep.append(plan)
                counts = {}
                for op in plan.model_structure.program.functions["main"].block.operations:
                    u = plan.get_compute_device_usage_for_mlprogram_operation(op)
                    if u is None: continue
                    d = type(u.preferred_compute_device).__name__
                    counts[d] = counts.get(d, 0) + 1
                plans[L] = counts
            out = {"op_placement_cpu_and_ne": plans}
        else:
            model, tok, _ = load_classifier("cpu"); model = model.float()
            enc = lambda p: tok(p, return_tensors="pt", add_special_tokens=True, max_length=512, truncation=True)
            ref = torch.load(f"{TMP}/ref.pt", weights_only=False)
            prompts, ref_logits = ref["prompts"], ref["ref"]
            r0 = rss(); t0 = time.perf_counter()
            mls = {L: ct.models.MLModel(p, compute_units=UNITS[name]) for L, p in paths.items()}
            keep.append(mls)
            out["load_s"] = time.perf_counter() - t0; out["rss_delta_mb"] = rss() - r0

            def run(p):
                e = enc(p); n = e["input_ids"].shape[1]
                L = next(b for b in sorted(mls) if b >= n)
                ids = np.zeros((1, L), np.int32); am = np.zeros((1, L), np.int32)
                ids[0, :n] = e["input_ids"][0].numpy(); am[0, :n] = 1
                o = mls[L].predict({"input_ids": ids, "attention_mask": am})
                return [torch.tensor(np.array(o[k])).reshape(1, -1).float() for k in sorted(o, key=lambda k: int(k.split("_")[1]))]
            ts, outs = [], []
            for i, p in enumerate(prompts[:5] + prompts):
                t0 = time.perf_counter(); o = run(p)
                if i >= 5: ts.append((time.perf_counter() - t0) * 1000); outs.append(o)
            out["latency"] = pct(ts)
            logits = [torch.cat([o[j] for o in outs]) for j in range(len(outs[0]))]
            X, ic = feature_rows(model.process_logits(logits), prompts)
            out["parity"] = {"per_head": [{"argmax_agree": float((logits[k].argmax(1) == ref_logits[k].argmax(1)).float().mean()),
                                           "max_abs_logit_diff": float((logits[k] - ref_logits[k]).abs().max())} for k in range(len(logits))],
                             "feature_X_max_abs_diff": float(np.abs(X - ref["Xr"]).max()),
                             "is_code_agree": float((ic == ref["cr"]).mean())}
    except Exception:
        out["error"] = traceback.format_exc()[-1500:]
    json.dump(out, open(f"{TMP}/{name}.json", "w"), default=str)
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    if "--arm" in sys.argv:
        arm(sys.argv[-1])
    try:
        main()
    finally:
        R.setdefault("latency", {})
        json.dump(R, open("results/ane_bench.json", "w"), indent=1, default=str)
