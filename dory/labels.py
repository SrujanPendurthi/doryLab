"""Loop-count labels: smallest k in KS at which Dory answers correctly."""
import argparse
import json
from pathlib import Path

from dory.config import KS
from dory.evalsets import load_split
from dory.sweep import latest_rows, read_rows


def min_k(correct_by_k):
    """Smallest correct k. None only when every k in KS is present and wrong."""
    for k in sorted(correct_by_k):
        if correct_by_k[k]:
            return k
    return None


def build_labels(sweep, items, variant="q5"):
    """One record per item with rows for `variant`; items with no rows are skipped."""
    by_id = {}
    for (i, k, v), r in latest_rows(read_rows(sweep)).items():
        if v == variant:
            by_id.setdefault(i, {})[k] = r
    out = []
    for it in items:
        rs = by_id.get(it.id)
        if not rs:
            continue
        ks = sorted(k for k in rs if k in KS)
        cbk = {k: bool(rs[k].get("correct")) and not rs[k].get("error") for k in ks}
        out.append({"id": it.id, "source": it.source, "prompt": it.prompt, "label_k": min_k(cbk),
                    "correct_by_k": cbk, "tokens_by_k": {k: [rs[k]["n_prompt"], rs[k]["n_gen"]] for k in ks}})
    return out


def class_balance(labels):
    ls = [r["label_k"] for r in labels if r["label_k"] is not None]
    return {k: ls.count(k) / len(ls) for k in sorted(set(ls))}


def non_monotone_rate(labels):
    """Fraction of full-grid items correct at some k but wrong at a larger k."""
    full = [r["correct_by_k"] for r in labels if all(k in r["correct_by_k"] for k in KS)]
    bad = sum(any(c[a] and not c[b] for a in KS for b in KS if b > a) for c in full)
    return bad / len(full) if full else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["holdout", "final", "train"])
    ap.add_argument("--sweep", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--variant", default="q5")
    a = ap.parse_args()
    items = load_split(a.split)
    labels = build_labels(a.sweep, items, a.variant)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in labels))
    print("n", len(labels), "class_balance", class_balance(labels))
    print("None", sum(r["label_k"] is None for r in labels), "skipped", len(items) - len(labels))
    if a.split != "train":
        print("non_monotone_rate", non_monotone_rate(labels))


if __name__ == "__main__":
    main()
