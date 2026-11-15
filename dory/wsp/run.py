"""Score a GGUF on an eval split (cached by weight hash) and apply the WSP accept/reject rules."""
import hashlib
import json
from pathlib import Path

from dory import evalsets
from dory.evaluate import paired_bootstrap
from dory.sweep import latest_rows, read_rows, run_sweep

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "results" / "wsp"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def evaluate(gguf: Path, split: str, k: int = 1, slots: int = 4) -> list[dict]:
    gguf = Path(gguf)
    sha = sha256_file(gguf)
    d = CACHE / sha[:12]
    d.mkdir(parents=True, exist_ok=True)
    meta_p = d / "meta.json"
    max_tokens = dict(evalsets.MAX_TOKENS)
    if meta_p.exists():
        if json.loads(meta_p.read_text())["max_tokens"] != max_tokens:
            raise RuntimeError(f"{meta_p}: cached under different MAX_TOKENS; delete the cache dir")
    else:
        meta_p.write_text(json.dumps({"sha256": sha, "gguf": str(gguf), "max_tokens": max_tokens}, indent=1))
    out = d / f"{split}_k{k}.jsonl"
    items = evalsets.load_split(split)
    run_sweep(items, [k], "wsp", gguf, out, slots)
    ids = {it.id for it in items}
    return [r for key, r in latest_rows(read_rows(out)).items() if key[2] == "wsp" and key[1] == k and key[0] in ids]


def _check(rows):
    for r in rows:
        if r.get("error"):
            raise ValueError(f"error row {r.get('id')}: {r['error']}")


def _mean(xs):
    return sum(xs) / len(xs)


def summarize(rows) -> dict:
    _check(rows)
    def stats(rs):
        return {"acc": _mean([bool(r["correct"]) for r in rs]), "n": len(rs),
                "mean_tokens": _mean([r["n_gen"] for r in rs])}
    out = stats(rows)
    out["trunc_rate"] = _mean([r["n_gen"] >= evalsets.MAX_TOKENS[r["source"]] for r in rows])
    out["per_source"] = {s: stats([r for r in rows if r["source"] == s]) for s in sorted({r["source"] for r in rows})}
    return out


def compare(base_rows, edit_rows) -> dict:
    _check(base_rows)
    _check(edit_rows)
    b = {r["id"]: r for r in base_rows}
    e = {r["id"]: r for r in edit_rows}
    if b.keys() != e.keys():
        raise ValueError(f"id mismatch: {len(b.keys() ^ e.keys())} ids on only one side")
    ids = sorted(b)
    d, lo, hi = paired_bootstrap([bool(e[i]["correct"]) for i in ids], [bool(b[i]["correct"]) for i in ids])
    per = {}
    for s in sorted({r["source"] for r in b.values()}):
        si = [i for i in ids if b[i]["source"] == s]
        per[s] = 100 * _mean([bool(e[i]["correct"]) - bool(b[i]["correct"]) for i in si])
    return {"dacc_pts": 100 * d, "ci_pts": (100 * lo, 100 * hi),
            "tokens_ratio": _mean([e[i]["n_gen"] for i in ids]) / _mean([b[i]["n_gen"] for i in ids]),
            "per_source_dacc_pts": per}


def screen_pass(base, edit) -> bool:
    _check(base)
    _check(edit)
    sb, se = summarize(base), summarize(edit)
    return (round(se["acc"] * se["n"]) >= round(sb["acc"] * sb["n"])
            and se["mean_tokens"] <= 1.05 * sb["mean_tokens"])


def accept(base_sel, edit_sel, ctrl_sel, base_arc, edit_arc) -> tuple[bool, str]:
    """base_sel = current best model on holdout; ctrl_sel = the edit's random control on holdout."""
    c = compare(base_sel, edit_sel)
    acc_path = c["dacc_pts"] >= 2.0 and c["ci_pts"][0] > 0
    tok_path = c["dacc_pts"] >= 0 and c["tokens_ratio"] <= 0.90
    if not (acc_path or tok_path):
        return False, "reject: no gain (needs dacc>=+2 with CI>0, or dacc>=0 with tokens<=0.90x)"
    cc = compare(base_sel, ctrl_sel)
    ctrl_tok = summarize(edit_sel)["mean_tokens"] <= 0.90 * summarize(ctrl_sel)["mean_tokens"]
    if not (c["dacc_pts"] - cc["dacc_pts"] >= 2.0 or (tok_path and ctrl_tok)):
        return False, "reject: control within 2 pts"
    worst = min(c["per_source_dacc_pts"].items(), key=lambda kv: kv[1])
    if worst[1] < -3.0:
        return False, f"reject: source {worst[0]} dropped {worst[1]:.1f} pts"
    arc = compare(base_arc, edit_arc)["dacc_pts"]
    if arc < -3.0:
        return False, f"reject: ARC dropped {arc:.1f} pts"
    return True, "accept: accuracy path" if acc_path else "accept: token path"
