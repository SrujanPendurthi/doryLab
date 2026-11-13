"""Trial one weight edit end to end: apply, quantize, screen, control, select, ledger."""
import argparse
import datetime
import json
import re
import shutil
import sys
import time
from pathlib import Path

from dory.wsp.edits import apply_edit, random_control, to_gguf
from dory.wsp.run import accept, compare, evaluate, screen_pass, summarize

ROOT = Path(__file__).resolve().parents[1]
BASE = {"model_dir": "models/bf16", "gguf": "models/gguf/dory-base-q8_0.gguf", "accepted": []}


def _gguf():
    try:
        import gguf
    except ImportError:
        sys.path.insert(0, str(ROOT / "vendor/llama.cpp/gguf-py"))
        import gguf
    return gguf


def gguf_name(hf_name: str) -> str:
    g = _gguf()
    # ponytail: 256 block cap, raise if a model ever has more layers
    return g.get_tensor_name_map(g.MODEL_ARCH.DORY, 256).get_name(hf_name, try_suffixes=(".weight",)) or hf_name


def gguf_diff_count(a: Path, b: Path) -> set:
    """Names of tensors in a whose raw bytes differ from (or are missing in) b."""
    import numpy as np
    g = _gguf()
    ta = {t.name: t for t in g.GGUFReader(str(a)).tensors}
    tb = {t.name: t for t in g.GGUFReader(str(b)).tensors}
    return {n for n, t in ta.items()
            if n not in tb or not np.array_equal(np.asarray(t.data).view(np.uint8), np.asarray(tb[n].data).view(np.uint8))}


def _rm(*paths):
    for p in paths:
        p = Path(p)
        shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)


def run_trial(edit: dict, root: Path = ROOT, allow_noop: bool = False) -> tuple[bool, str]:
    root = Path(root)
    ledger, state = root / "research/wsp/ledger.md", root / "research/wsp/best.json"
    text = ledger.read_text() if ledger.exists() else ""
    eid = edit["id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", eid):
        raise SystemExit(f"bad edit id {eid!r}")
    if re.search(rf"^## {re.escape(eid)}$", text, re.M):
        raise SystemExit(f"edit {eid} already in ledger")
    best = json.loads(state.read_text()) if state.exists() else BASE
    best_dir, best_gguf = root / best["model_dir"], root / best["gguf"]
    models = root / "results/wsp/models"
    mdir, mgguf = models / eid, models / f"{eid}.gguf"
    cdir, cgguf, cnpz = models / f"{eid}_ctrl", models / f"{eid}_ctrl.gguf", models / f"{eid}_ctrl_npz"
    t0 = time.time()
    models.mkdir(parents=True, exist_ok=True)
    info, extra, ok = {}, {}, False
    try:
        st = apply_edit(edit, best_dir, mdir)
        info = {"n_requested": st.n_requested, "n_changed": st.n_changed, "tensors": st.tensors}
        if st.n_changed == 0 and not allow_noop:
            reason = "no-op edit"
        else:
            to_gguf(mdir, mgguf)
            diff = gguf_diff_count(mgguf, best_gguf)
            expected = {gguf_name(n) for n in st.tensors if st.delta_l2.get(n, 0) > 0}
            info["gguf_tensors_differing"] = len(diff)
            if expected - diff:
                reason = "edit lost in quantization"
            else:
                sb, se = evaluate(best_gguf, "wsp_screen"), evaluate(mgguf, "wsp_screen")
                extra["screen"] = {"best": summarize(sb), "edit": summarize(se)}
                if not screen_pass(sb, se):
                    reason = "reject: screen"
                else:
                    ctrl = random_control(edit, best_dir, seed=0, npz_dir=cnpz)
                    apply_edit(ctrl, best_dir, cdir)
                    to_gguf(cdir, cgguf)
                    bsel, esel, csel = (evaluate(g, "holdout") for g in (best_gguf, mgguf, cgguf))
                    barc, earc = evaluate(best_gguf, "arc"), evaluate(mgguf, "arc")
                    extra["edit_vs_best"], extra["ctrl_vs_best"] = compare(bsel, esel), compare(bsel, csel)
                    extra["arc_dacc_pts"] = compare(barc, earc)["dacc_pts"]
                    ok, reason = accept(bsel, esel, csel, barc, earc)
        state.parent.mkdir(parents=True, exist_ok=True)
        if ok:
            state.write_text(json.dumps({"model_dir": str(mdir.relative_to(root)), "gguf": str(mgguf.relative_to(root)),
                                         "accepted": best["accepted"] + [eid]}, indent=1))
    except BaseException as e:
        ok, reason = False, f"ERROR {type(e).__name__}: {e}"
        _write_ledger(ledger, text, edit, info, extra, ok, reason, time.time() - t0)
        raise
    finally:
        _rm(cdir, cgguf, cnpz)
        if not ok:
            _rm(mdir, mgguf)
    _write_ledger(ledger, text, edit, info, extra, ok, reason, time.time() - t0)
    return ok, reason


def _write_ledger(ledger, text, edit, info, extra, ok, reason, secs):
    hours = sum(float(h) for h in re.findall(r"<!-- hours: ([\d.]+) -->", text)) + secs / 3600
    decision = "ERROR" if reason.startswith("ERROR") else "ACCEPT" if ok else "REJECT"
    entry = (f"\n## {edit['id']}\n\n- time: {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}\n"
             f"- hypothesis: {edit.get('hypothesis', '')}\n- evidence: {json.dumps(edit.get('evidence', []))}\n"
             f"- predicted: {edit.get('predict')}\n- realized: {json.dumps(info)}\n"
             + "".join(f"- {k}: {json.dumps(v, default=str)}\n" for k, v in extra.items())
             + f"- decision: {decision} ({reason})\n- wall-clock: {secs:.1f}s\n"
             f"- cumulative: {hours:.3f}h\n<!-- hours: {secs / 3600:.6f} -->\n")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger, "a") as f:
        f.write(entry)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("edit")
    ap.add_argument("--allow-noop", action="store_true")
    a = ap.parse_args()
    edit = json.loads(Path(a.edit).read_text())
    ok, reason = run_trial(edit, ROOT, a.allow_noop)
    print(f"WSP_TRY {edit['id']} {'ACCEPT' if ok else 'REJECT'} {reason}")


if __name__ == "__main__":
    main()
