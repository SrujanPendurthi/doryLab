"""Paper numbers and tables: every result macro comes from results/ files (run: python -m scripts.paper.numbers).

Text-policy rows (holdout, final) are recomputed from the saved post-regrade policy: holdout from
results/features_text_holdout.npy, final by running the classifier on CPU (cached to features_text_final.npy).
Their seconds are modeled seconds plus the measured live MPS controller overhead (results/live_gate.json).
Only the probe row (needs a live probe server) and a few other values are pinned from results/decisions.md
and marked `decisions.md` beside the macro.
"""
import json
import subprocess
from pathlib import Path

import numpy as np

from dory.bench import load_speed
from dory.evalsets import MAX_TOKENS
from dory.evaluate import _correct, fixed, oracle, paired_bootstrap, policy_metrics, random_matched
from dory.sweep import latest_rows, read_rows

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
SRC = {"gsm8k": "Gsm", "mmlu_pro": "Mmlu", "mbpp": "Mbpp"}
KW = {1: "One", 2: "Two", 4: "Four"}
SRC_LABEL = {"gsm8k": "GSM8K", "mmlu_pro": "MMLU-Pro", "mbpp": "MBPP"}


def _labels(split):
    return [json.loads(l) for l in (R / f"labels_{split}.jsonl").read_text().splitlines() if l.strip()]


PRE_REGRADE_COMMIT = "78778a0"  # "feat: Phase 1 loop-count labels", before scripts/regrade.py


def _pre_labels(split):
    out = subprocess.run(["git", "show", f"{PRE_REGRADE_COMMIT}:results/labels_{split}.jsonl"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    return [json.loads(l) for l in out.splitlines() if l.strip()]


def _feat(name):
    """Cached feature file: worktree results/, else the main checkout's (gitignored, not in worktrees)."""
    for d in (R, ROOT.parents[1] / "results"):
        if (d / name).exists():
            return d / name
    return R / name


def _text_row(split, L, speed, overhead_s):
    """Post-regrade text policy on `split`: choice by saved joblib (as scripts/final_test.py), tau unchanged.
    Seconds: modeled + measured live MPS overhead (CPU classifier timing is not comparable)."""
    from dory.policy import choose_k, load_policy, proba_for_ks
    f = _feat(f"features_text_{split}.npy")
    if f.exists() and len(X := np.load(f)) == len(L):
        pass
    else:
        from dory.features import text_features
        X = text_features([r["prompt"] for r in L], device="cpu")[0]
        np.save(R / f"features_text_{split}.npy", X)
    pol = load_policy(_feat("policy_text.joblib"))
    ks = choose_k(proba_for_ks(pol["model"], X), X[:, -1].astype(bool), pol["tau"])  # last column = is_code
    return {r["id"]: int(k) for r, k in zip(L, ks)}, {r["id"]: overhead_s for r in L}


def _latest(path):
    """Rows of a cfg/id result file with latest_rows semantics (error rows skipped, latest retry wins)."""
    rows = [{**r, "k": r["cfg"], "variant": None} for r in read_rows(path)]
    return list(latest_rows(rows).values())


def _floor(L):
    """Fixed k=1 plus a code floor (k=2 for MBPP). Found after seeing holdout: post hoc."""
    return {r["id"]: (2 if r["source"] == "mbpp" else 1) for r in L}


def _gm(x):
    return float(np.exp(np.mean(np.log(x))))


def _ci(ratios, n=10000, seed=0):
    lr = np.log(ratios)
    idx = np.random.default_rng(seed).integers(0, len(lr), size=(n, len(lr)))
    return tuple(float(np.exp(np.percentile(lr[idx].mean(1), q))) for q in (2.5, 97.5))


def _policies(L):
    return {"Fixed k=1": fixed(L, 1), "Fixed k=2": fixed(L, 2), "Fixed k=4": fixed(L, 4),
            "Fixed k=1 + code floor": _floor(L), "Per-prompt oracle": oracle(L)}


def _rows(L, pols, speed, extra=None):
    """name -> metrics incl. delta vs fixed2 with paired-bootstrap CI."""
    base = [_correct(r, 2) for r in L]
    out = {}
    for name, ch in pols.items():
        m = policy_metrics(L, ch, speed, (extra or {}).get(name))
        m["d"], m["lo"], m["hi"] = paired_bootstrap([_correct(r, ch[r["id"]]) for r in L], base)
        out[name] = m
    return out


def _table(rows, extra=()):
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Policy & Acc. & Mean $k$ & Mean s & $\Delta$ vs $k{=}2$ [95\% CI] \\", r"\midrule"]
    for name, m in list(rows.items()) + list(extra):
        lines.append(f"{name} & {m['acc']:.3f} & {m['mean_k']:.2f} & {m['mean_seconds']:.2f} & "
                     f"{m['d']:+.3f} [{m['lo']:+.3f}, {m['hi']:+.3f}] \\\\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def _trunc(L):
    """Train labels are lazy (k>1 only for items wrong at k=1), so train k=2/4 rows are a biased subset.
    (source, k) -> n, truncation rate, accuracy among non-truncated, accuracy overall."""
    out = {}
    for s in SRC:
        for k in (1, 2, 4):
            rs = [r for r in L if r["source"] == s and str(k) in {str(x) for x in r["tokens_by_k"]}]
            if not rs:
                continue
            tr = [r["tokens_by_k"][str(k)][1] >= MAX_TOKENS[s] for r in rs]
            ok = [_correct(r, k) for r in rs]
            nt = [o for o, t in zip(ok, tr) if not t]
            out[(s, k)] = {"n": len(rs), "trunc": float(np.mean(tr)), "n_trunc": int(sum(tr)),
                           "acc_nt": float(np.mean(nt)) if nt else float("nan"), "acc": float(np.mean(ok))}
    return out


def compute():
    speed = load_speed(R / "speed.json")
    V = {}

    def put(name, val, nd=3):
        assert name.isalpha(), name
        V[name] = val if isinstance(val, str) else f"{val:.{nd}f}"

    sp = {}
    for split in ("holdout", "final", "train"):
        L = _labels(split)
        sp[split] = L
        put(f"{split.capitalize()}N", len(L), 0)
    T = {}

    # --- controllers on holdout and final (post-regrade labels) ---
    names = {"Fixed k=1": "FixedOne", "Fixed k=2": "FixedTwo", "Fixed k=4": "FixedFour",
             "Fixed k=1 + code floor": "Floor", "Per-prompt oracle": "Oracle"}
    rows = {}
    for split in ("holdout", "final"):
        L, S = sp[split], split.capitalize()
        rows[split] = _rows(L, _policies(L), speed)
        for name, m in rows[split].items():
            p = S + names[name]
            put(p + "Acc", m["acc"]); put(p + "MeanK", m["mean_k"], 2); put(p + "Sec", m["mean_seconds"], 2)
            put(p + "Delta", m["d"]); put(p + "Lo", m["lo"]); put(p + "Hi", m["hi"])
    gate = json.loads((R / "live_gate.json").read_text())
    ov = gate["overhead_ms_text_policy"] / 1000
    # text policy: acc/k/CI from the saved post-regrade joblib; seconds = modeled + measured live MPS overhead
    for split in ("holdout", "final"):
        L, S = sp[split], split.capitalize()
        ch, ex = _text_row(split, L, speed, ov)
        pols = {**_policies(L), "Text policy": ch}
        m = _rows(L, pols, speed, {"Text policy": ex})["Text policy"]
        rows[split]["Text policy"] = m
        put(S + "TextAcc", m["acc"]); put(S + "TextMeanK", m["mean_k"], 2); put(S + "TextSec", m["mean_seconds"], 2)
        put(S + "TextDelta", m["d"]); put(S + "TextLo", m["lo"]); put(S + "TextHi", m["hi"])
    rm = random_matched(sp["holdout"], 2.0)
    put("HoldoutRandomAcc", policy_metrics(sp["holdout"], rm, speed)["acc"])
    f2, f1 = rows["final"]["Fixed k=2"], rows["final"]["Fixed k=1 + code floor"]
    put("FloorLatencyRatio", f1["mean_seconds"] / f2["mean_seconds"], 2)  # C3: ~62%
    # chosen policy = fixed k=2 (its code floor is k=2 already): identical to Fixed k=2
    put("FinalChosenAcc", f2["acc"]); put("FinalChosenSec", f2["mean_seconds"], 2)

    # widest paired-bootstrap half-width among non-degenerate policies (fixed 2 and the oracle excluded)
    for split in ("holdout", "final"):
        hw = max((m["hi"] - m["lo"]) / 2 for n, m in rows[split].items() if n not in ("Fixed k=2", "Per-prompt oracle"))
        put(split.capitalize() + "CiHalf", hw)

    # oracle headroom: k=1 already right vs rescued by k=2/4, unrescuable
    for split in ("holdout", "final"):
        L, S = sp[split], split.capitalize()
        lk = [r["label_k"] for r in L]
        put(S + "OracleAlready", np.mean([k == 1 for k in lk])); put(S + "OracleRescued", np.mean([k in (2, 4) for k in lk]))
        put(S + "OracleNever", np.mean([k is None for k in lk]))
        put(S + "OracleRescuedN", sum(k in (2, 4) for k in lk), 0)

    # --- learned controllers: pinned from decisions.md (need classifier / live probe server) ---
    # decisions.md "Task 8 (rerun after regrade)" and "Probe holdout report (rerun after regrade)"
    pinned = {"HoldoutProbeAcc": "0.590", "HoldoutProbeDelta": "+0.003", "HoldoutProbeLo": "-0.023",
              "HoldoutProbeHi": "+0.033", "HoldoutProbeSec": "18.13",
              # pre-regrade Task 8 text: +0.007 [-0.010, +0.027]
              "HoldoutTextPreDelta": "+0.007", "HoldoutTextPreLo": "-0.010", "HoldoutTextPreHi": "+0.027",
              # decisions.md Task 10 table (pre-regrade text on final, tau 0.4326)
              "FinalPreTextAcc": "0.630", "FinalPreTextMeanK": "1.94", "FinalPreTextSec": "20.43",
              "FinalPreTextDelta": "+0.010", "FinalPreTextLo": "-0.020", "FinalPreTextHi": "+0.040"}
    V.update(pinned)
    # Both splits were graded twice (Ruling AJ: grader made truncation-aware). Pre-regrade labels are the
    # Phase 1 commit; they are recomputed from git, not typed in.
    for split in ("holdout", "final"):
        L, S = _pre_labels(split), split.capitalize() + "Pre"
        for name, m in _rows(L, _policies(L), speed).items():
            put(S + names[name] + "Acc", m["acc"]); put(S + names[name] + "Sec", m["mean_seconds"], 2)
    V["FinalPreChosenAcc"] = V["FinalPreFixedTwoAcc"]

    # --- truncation confound (new analysis) ---
    for split in ("holdout", "train"):
        T[split] = _trunc(sp[split])
        for (s, k), t in T[split].items():
            p = split.capitalize() + SRC[s] + "K" + KW[k]
            put(p + "TruncRate", t["trunc"]); put(p + "AccNT", t["acc_nt"]); put(p + "N", t["n"], 0)
            put(p + "TruncN", t["n_trunc"], 0)

    # --- systems ---
    by = {}
    for r in _latest(R / "spec_ab.jsonl"):
        by.setdefault(r["cfg"], {})[r["id"]] = r
    sysrows = {}
    for cfg, S in (("n4", "Four"), ("n6", "Six")):
        ids = sorted(by[cfg])
        ratio = [by[cfg][i]["gen_tps"] / by["plain"][i]["gen_tps"] for i in ids]
        lo, hi = _ci(ratio)
        acc = sum(by[cfg][i]["accepted"] for i in ids) / sum(by[cfg][i]["draft_n"] for i in ids)
        put(f"SpecN{S}Speedup", _gm(ratio), 2); put(f"SpecN{S}Lo", lo, 2); put(f"SpecN{S}Hi", hi, 2)
        put(f"SpecN{S}Accept", acc, 2); put(f"SpecN{S}Tps", np.mean([by[cfg][i]["gen_tps"] for i in ids]), 2)
        put(f"SpecN{S}Faster", sum(x > 1 for x in ratio), 0)
        sysrows[cfg] = (_gm(ratio), lo, hi, acc)
        same = [by[cfg][i]["text"] == by["plain"][i]["text"] for i in ids]
        put(f"SpecN{S}Identical", sum(same), 0)
    put("SpecPlainTps", np.mean([r["gen_tps"] for r in by["plain"].values()]), 2)
    put("SpecPrompts", len(by["plain"]), 0)
    k4 = json.loads((R / "spec_k4.json").read_text())
    put("SpecSequentialRatio", k4["speedup"], 2)  # sequential-arm artifact (C6)
    put("SpecSequentialAccept", k4["acceptance"], 2)
    off = {}
    for r in _latest(R / "offload_ab.jsonl"):
        off.setdefault(r["cfg"], {})[r["id"]] = r["gen_tps"]
    ratio = [off["plain_ngl54"][i] / off["plain"][i] for i in off["plain"]]
    put("OffloadSpeedup", _gm(ratio), 2)
    # decisions.md Task 3: llama-server n-gram lookup, 10 MBPP prompts, 16.33 vs 12.08 tok/s (no results file)
    put("NgramSpeedup", 12.08 / 16.33, 2)
    q8 = {i: r["correct"] for (i, _, _), r in latest_rows(read_rows(R / "sweep_holdout_q8.jsonl")).items()}
    f16 = {r["id"]: _correct(r, 4) for r in sp["holdout"] if r["source"] != "mbpp"}
    # q8 file covers gsm8k+mmlu_pro (200 items); regraded labels give the f16 side
    put("QEightN", len(q8), 0)
    put("QEightDelta", (sum(q8.values()) - sum(f16[i] for i in q8)) / len(q8), 3)
    put("QEightAcc", np.mean(list(q8.values()))); put("QSixteenAcc", np.mean([f16[i] for i in q8]))
    ane = json.loads((R / "ane_bench.json").read_text())
    put("AneMs", ane["coreml"]["CPU_AND_NE"]["latency"]["median_ms"], 1)
    put("MpsMs", ane["latency"]["pytorch_mps"]["median_ms"], 1)
    put("CpuMs", ane["latency"]["pytorch_cpu"]["median_ms"], 1)
    put("AneSpeedup", ane["latency"]["pytorch_mps"]["median_ms"] / ane["coreml"]["CPU_AND_NE"]["latency"]["median_ms"], 1)
    put("LiveOverheadMs", gate["overhead_ms_live_none"], 1)
    put("LiveGateN", gate["n"], 0)
    # KV memory: RSS of the k=4, slots=1 server, f16 vs q8_0 (results/bench.csv; decisions.md Task 3: 9.6 vs 8.0 GB)
    import csv
    rss = {r["kv"]: float(r["rss_mb"]) for r in csv.DictReader(open(R / "bench.csv")) if r["k"] == "4" and r["slots"] == "1"}
    put("KvSavingGb", (rss["f16"] - rss["q8_0"]) / 1024, 1)

    compute.tables = (rows, T, sysrows, sp, speed)
    return V


def write(out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    V = compute()
    rows, T, sysrows, sp, speed = compute.tables
    (out_dir / "numbers.json").write_text(json.dumps(V, indent=1, sort_keys=True) + "\n")
    (out_dir / "numbers.tex").write_text("% generated by scripts/paper/numbers.py; do not edit\n" + "".join(
        f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(V.items())))
    # probe row needs a live probe server: pinned from decisions.md
    t = lambda v: float(V[v])
    probe = [("Hidden-state probe (matched $k{=}2$)", {"acc": t("HoldoutProbeAcc"), "mean_k": 2.0,
             "mean_seconds": t("HoldoutProbeSec"), "d": t("HoldoutProbeDelta"), "lo": t("HoldoutProbeLo"), "hi": t("HoldoutProbeHi")})]
    (out_dir / "tab_controllers_holdout.tex").write_text(_table(rows["holdout"], probe))
    (out_dir / "tab_controllers_final.tex").write_text(_table(rows["final"]))

    # per source: acc / mean s on holdout for each policy
    pols, L = _policies(sp["holdout"]), sp["holdout"]
    lines = [r"\begin{tabular}{l" + "rr" * len(pols) + "}", r"\toprule",
             "Source & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{n}}}" for n in pols) + r" \\",
             " & " + " & ".join("Acc. & s" for _ in pols) + r" \\", r"\midrule"]
    for s, lab in SRC_LABEL.items():
        sub = [r for r in L if r["source"] == s]
        cells = []
        for ch in pols.values():
            m = policy_metrics(sub, ch, speed)
            cells.append(f"{m['acc']:.3f} & {m['mean_seconds']:.1f}")
        lines.append(f"{lab} ($n{{=}}{len(sub)}$) & " + " & ".join(cells) + r" \\")
    (out_dir / "tab_per_source.tex").write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""]))

    sysl = [r"\begin{tabular}{lrl}", r"\toprule", r"Configuration & Speed vs plain & Notes \\", r"\midrule"]
    for cfg, lab in (("n4", "Self-speculative, draft 4"), ("n6", "Self-speculative, draft 6")):
        g, lo, hi, acc = sysrows[cfg]
        sysl.append(f"{lab} & {g:.2f}$\\times$ [{lo:.2f}, {hi:.2f}] & acceptance {acc:.2f}, outputs identical \\\\")
    sysl += [f"n-gram lookup (MBPP) & {V['NgramSpeedup']}$\\times$ & \\\\",
             f"Partial GPU offload (54/72 layers) & {V['OffloadSpeedup']}$\\times$ & same graded accuracy \\\\",
             f"Sequential A/B (artifact) & {V['SpecSequentialRatio']}$\\times$ & arms hours apart under varying load \\\\",
             f"Core ML (Neural Engine) classifier & {V['AneSpeedup']}$\\times$ & {V['AneMs']} vs {V['MpsMs']} ms (MPS) \\\\",
             r"\bottomrule", r"\end{tabular}", ""]
    (out_dir / "tab_systems.tex").write_text("\n".join(sysl))

    tl = [r"\begin{tabular}{llrrrr}", r"\toprule",
          r"Split & Source, $k$ & $n$ & Truncated & Acc. (non-trunc.) & Acc. (all) \\", r"\midrule"]
    for split in ("holdout", "train"):
        for (s, k), v in T[split].items():
            tl.append(f"{split} & {SRC_LABEL[s]}, {k} & {v['n']} & {v['trunc']:.3f} & {v['acc_nt']:.3f} & {v['acc']:.3f} \\\\")
    (out_dir / "tab_truncation.tex").write_text("\n".join(tl + [r"\bottomrule", r"\end{tabular}", ""]))


if __name__ == "__main__":
    write(ROOT / "paper" / "generated")
