"""Re-grade sweep jsonl files offline with truncation-aware graders: python -m scripts.regrade FILE... [--split S]"""
import argparse
import json
import os
from pathlib import Path

from dory.evalsets import MAX_TOKENS, grade, load_split


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--split", default=None, help="default: inferred from sweep_<split>[_x].jsonl")
    a = ap.parse_args(argv)
    for p in a.files:
        split = a.split or p.stem.split("_")[1]
        items = {it.id: it for it in load_split(split)}
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        up = down = 0
        for r in rows:
            if r.get("error") or r["source"] == "mbpp":  # truncation never changes mbpp; rerunning code could flake
                continue
            new = bool(grade(items[r["id"]], r["output"], r["n_gen"] >= MAX_TOKENS[r["source"]]))
            up += new and not r["correct"]
            down += r["correct"] and not new
            r["correct"] = new
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
        os.replace(tmp, p)
        print(f"{p}: False->True {up}, True->False {down}")


if __name__ == "__main__":
    main()
