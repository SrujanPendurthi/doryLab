"""Fixed-k sweep: label how many loops each prompt needs. Rows append to JSONL; reruns resume."""
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dory.client import ServerError, complete
from dory.config import KS
from dory.evalsets import MAX_TOKENS, grade, load_split
from dory.server import Server, free_port, server_args

ROOT = Path(__file__).resolve().parent.parent


def read_rows(path: Path) -> list[dict]:
    rows = []
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if isinstance(r, dict):
                rows.append(r)
    return rows


def latest_rows(rows):
    """(id, k, variant) -> last non-error row, else last row."""
    out = {}
    for r in rows:
        key = (r.get("id"), r.get("k"), r.get("variant"))
        if r.get("error") and key in out and not out[key].get("error"):
            continue
        out[key] = r
    return out


def _final(r):
    e = r.get("error")
    return not e or e.startswith("grade:") or "HTTP 4" in e  # others (timeout, 5xx) are retried


def pending(items, k, variant, done, lazy):
    mine = {key: r for key, r in latest_rows(done).items() if key[2] == variant}
    have = {i for (i, kk, _), r in mine.items() if kk == k and _final(r)}
    if lazy:
        have |= {i for (i, kk, _), r in mine.items() if kk < k and r.get("correct") is True}
    return [it for it in items if it.id not in have]


def run_sweep(items, ks, variant, gguf, out, slots, kv_type="f16", lazy=False, max_tokens=None):
    caps = max_tokens or MAX_TOKENS
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    torn = out.exists() and out.stat().st_size > 0 and not out.read_bytes().endswith(b"\n")
    with out.open("a") as f:
        if torn:
            f.write("\n")
        def write(r):
            with lock:
                f.write(json.dumps(r) + "\n")
                f.flush()

        for k in sorted(ks):
            todo = pending(items, k, variant, read_rows(out), lazy)
            if not todo:
                continue
            port = free_port()
            with Server(server_args(gguf, k, port, slots=slots, kv_type=kv_type), port) as srv:
                def one(it):
                    r = {"id": it.id, "source": it.source, "k": k, "variant": variant, "output": "",
                         "correct": False, "n_prompt": 0, "n_gen": 0, "error": None}
                    try:
                        c = complete(srv.url, it.prompt, caps[it.source])
                        r.update(output=c.text, n_prompt=c.n_prompt, n_gen=c.n_gen)
                        try:
                            r["correct"] = bool(grade(it, c.text, c.n_gen >= caps[it.source]))
                        except Exception as e:
                            r["error"] = f"grade: {e}"
                    except ServerError as e:
                        r["error"] = str(e)
                    write(r)

                with ThreadPoolExecutor(slots) as ex:
                    list(ex.map(one, todo))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["holdout", "final", "train"])
    ap.add_argument("--ks", default=",".join(map(str, KS)))
    ap.add_argument("--variant", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--kv", default="f16")
    ap.add_argument("--lazy", action="store_true")
    ap.add_argument("--sources")
    ap.add_argument("--max-tokens", type=int, help="override cap for every selected source")
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--gguf", type=Path, default=ROOT / "models/dory-q5.gguf")
    a = ap.parse_args()
    items = load_split(a.split)
    if a.sources:
        keep = set(a.sources.split(","))
        items = [i for i in items if i.source in keep]
    mt = {s: a.max_tokens for s in {i.source for i in items}} if a.max_tokens else None
    ks = [int(x) for x in a.ks.split(",")]
    run_sweep(items, ks, a.variant, a.gguf, a.out, a.slots, a.kv, a.lazy, mt)
    if mt:  # per-k summary
        cap = a.max_tokens
        for k in ks:
            rs = [r for r in latest_rows(read_rows(a.out)).values() if r["variant"] == a.variant and r["k"] == k]
            n = max(len(rs), 1)
            print(f"k={k} n={len(rs)} acc={sum(bool(r['correct']) for r in rs) / n:.3f} "
                  f"trunc={sum(r['n_gen'] >= cap for r in rs) / n:.3f}")


if __name__ == "__main__":
    main()
