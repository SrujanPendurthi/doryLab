"""Paper figures (vector PDF). Values come from paper/generated/numbers.json (scripts/paper/numbers.py) and results/.
Run: python -m scripts.paper.figures. Palette: dataviz skill reference light slots, fixed order, shape + label as
secondary encoding.
Sequential-arm throughput is load-confounded (see C6), so only acceptance is shown from the sweep."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dory.sweep import latest_rows, read_rows
from scripts.paper.numbers import KW, ROOT, SRC, SRC_LABEL, _labels, _latest, write

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
INK, INK2, MUTED, GRID, GRAY = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRAY, "axes.labelcolor": INK2, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
                     "pdf.fonttype": 42, "legend.frameon": False, "figure.dpi": 100})


def _num():
    p = ROOT / "paper" / "generated" / "numbers.json"
    if not p.exists():
        write(p.parent)
    return {k: float(v) for k, v in json.loads(p.read_text()).items() if _isnum(v)}


def _isnum(v):
    try:
        float(v)
        return True
    except ValueError:
        return False


def fig_pareto(out, V):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2))
    handles = {}
    for ax, S, title in zip(axes, ("Holdout", "Final"), ("Holdout (n=300)", "Final test (n=100)")):
        g = lambda n, f: V[f"{S}{n}{f}"]
        fx = [(g(n, "Sec"), g(n, "Acc")) for n in ("FixedOne", "FixedTwo", "FixedFour")]
        ax.plot(*zip(*fx), "-", color=BLUE, lw=1.5, zorder=2)
        handles["fixed $k$ = 1, 2, 4"] = ax.scatter(*zip(*fx), color=BLUE, marker="o", s=36, zorder=3, edgecolor="white", linewidth=1)
        for (x, y), k in zip(fx, (1, 2, 4)):
            ax.annotate(str(k), (x, y), (5, -9), textcoords="offset points", fontsize=7.5, color=INK2)
        pts = [("Floor", "fixed 1 + code floor", ORANGE, "s"), ("Text", "text policy", AQUA, "^"),
               ("Oracle", "per-prompt oracle", MAGENTA, "D")]
        if S == "Holdout":
            pts.insert(2, ("Probe", "hidden-state probe", YELLOW, "v"))
        for n, lab, c, m in pts:
            handles[lab] = ax.scatter([g(n, "Sec")], [g(n, "Acc")], color=c, marker=m, s=40, zorder=3, edgecolor="white", linewidth=1)
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel("mean modeled seconds per prompt")
        ax.set_xlim(0, ax.get_xlim()[1] * 1.05)
    axes[0].set_ylabel("accuracy")
    order = ["fixed $k$ = 1, 2, 4", "fixed 1 + code floor", "text policy", "hidden-state probe", "per-prompt oracle"]
    fig.legend([handles[o] for o in order], order, loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(out / "fig_pareto.pdf")


def fig_headroom(out):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.9))
    cols = [(BLUE, "right at $k$=1"), (AQUA, "rescued by $k$=2 or 4"), (GRAY, "wrong at all $k$")]
    for ax, split, title in zip(axes, ("holdout", "final"), ("Holdout", "Final test")):
        L = _labels(split)
        for i, s in enumerate(SRC):
            lk = [r["label_k"] for r in L if r["source"] == s]
            fr = [np.mean([k == 1 for k in lk]), np.mean([k in (2, 4) for k in lk]), np.mean([k is None for k in lk])]
            left = 0
            for (c, _), f in zip(cols, fr):
                ax.barh(i, f, left=left, color=c, height=0.6, edgecolor="white", linewidth=1.5)
                if f >= 0.10:
                    ax.text(left + f / 2, i, f"{f:.0%}", ha="center", va="center", fontsize=7.5,
                            color="white" if c == BLUE else INK)
                left += f
        ax.set_yticks(range(3), [f"{SRC_LABEL[s]}\n(n={sum(r['source'] == s for r in L)})" for s in SRC])
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
        ax.set_xlabel("share of prompts")
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", visible=False)
    axes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c, _ in cols], labels=[l for _, l in cols],
                   loc="lower center", bbox_to_anchor=(1.1, -0.45), ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_headroom.pdf", bbox_inches="tight")


def fig_truncation(out, V):
    """Holdout only: train k=2/4 cells cover only items that failed at k=1 (lazy labels)."""
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5), sharex=True)
    cols = {1: BLUE, 2: ORANGE, 4: AQUA}
    w = 0.26
    for ax, key, lab in zip(axes, ("TruncRate", "AccNT"), ("truncated at token cap", "accuracy, non-truncated outputs")):
        for j, k in enumerate((1, 2, 4)):
            vals = [V[f"Holdout{SRC[s]}K{KW[k]}{key}"] for s in SRC]
            xs = np.arange(3) + (j - 1) * w
            ax.bar(xs, vals, w * 0.88, color=cols[k], label=f"$k$={k}")
            for x, v in zip(xs, vals):
                ax.text(x, v + 0.01, f"{v:.0%}", ha="center", fontsize=6.5, color=INK2)
        ax.set_xticks(range(3), [SRC_LABEL[s] for s in SRC])
        ax.set_ylabel(lab)
        ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.12)
    axes[0].legend(fontsize=8, title="loops", title_fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fig_truncation.pdf")


def spec_data():
    R = ROOT / "results"
    sweep = {}
    for r in _latest(R / "draft_sweep.jsonl"):
        sweep.setdefault(r["cfg"], []).append(r)
    by = {}
    for r in _latest(R / "spec_ab.jsonl"):
        by.setdefault(r["cfg"], {})[r["id"]] = r["gen_tps"]
    ratios = {c: np.array([by[c][i] / by["plain"][i] for i in sorted(by[c])]) for c in ("n4", "n6")}
    return sweep, ratios


def fig_spec(out, V):
    sweep, ratios = spec_data()
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.7), gridspec_kw={"width_ratios": [1.15, 1]})
    cfgs = [c for c in ("n2", "n4", "n6", "n8", "n8_p07") if c in sweep]
    acc = [sum(r["accepted"] for r in sweep[c]) / sum(r["draft_n"] for r in sweep[c]) for c in cfgs]
    a.bar(range(len(cfgs)), acc, 0.6, color=BLUE)
    for i, v in enumerate(acc):
        a.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=7.5, color=INK2)
    a.set_xticks(range(len(cfgs)), [c[1:].replace("_p07", "\n(p07)") for c in cfgs])
    a.set_ylim(0, 1.05)
    a.set_xlabel("max draft tokens $n_{max}$ (self-speculative)")
    a.set_ylabel("draft acceptance rate")
    a.set_title("Draft-length sweep: acceptance", fontsize=9)
    a.grid(axis="x", visible=False)
    rng = np.random.default_rng(0)
    for i, (c, col) in enumerate((("n4", BLUE), ("n6", ORANGE))):
        r = ratios[c]
        b.scatter(i + rng.uniform(-0.12, 0.12, len(r)), r, s=14, color=col, alpha=0.8, edgecolor="white", linewidth=0.5, zorder=3)
        gm = float(np.exp(np.mean(np.log(r))))
        b.hlines(gm, i - 0.25, i + 0.25, color=INK, lw=1.8, zorder=4)
        b.text(i + 0.28, gm, f"{gm:.2f}x", va="center", fontsize=8)
    b.axhline(1, color=MUTED, lw=0.8, ls="--")
    b.axhline(V["SpecSequentialRatio"], color=MAGENTA, lw=1.5)
    b.text(1.45, V["SpecSequentialRatio"] + 0.03, f"sequential A/B: {V['SpecSequentialRatio']:.2f}x\n(artifact)", ha="right", va="bottom",
           fontsize=7.5, color=INK2)
    b.set_xticks([0, 1], ["draft 4", "draft 6"])
    b.set_xlim(-0.5, 1.5)
    b.set_ylim(0, None)
    b.set_ylabel("speed vs plain, same prompt (x)")
    b.set_title(f"Interleaved A/B per prompt (n={len(ratios['n4'])})", fontsize=9)
    b.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out / "fig_spec.pdf")


def make_all(out=ROOT / "paper" / "figures"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    V = _num()
    fig_pareto(out, V)
    fig_headroom(out)
    fig_truncation(out, V)
    fig_spec(out, V)
    plt.close("all")


if __name__ == "__main__":
    make_all()
