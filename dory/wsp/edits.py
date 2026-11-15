"""Reversible direct weight edits on a BF16 safetensors snapshot, plus size-matched random controls."""
import hashlib
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

REPO = Path(__file__).resolve().parents[2]


@dataclass
class EditStats:
    n_requested: int
    n_changed: int
    tensors: list
    delta_l2: dict


def _check(idx, shape):
    if len(idx) != len(shape) or any(not 0 <= i < s for i, s in zip(idx, shape)):
        raise IndexError(f"index {list(idx)} out of range for shape {tuple(shape)}")
    return tuple(idx)


def _apply_op(w: torch.Tensor, op: dict) -> int:
    """Apply op to float32 tensor w in place; return number of requested elements."""
    kind = op["op"]
    if kind in ("set", "add", "scale"):
        idx = [_check(i, w.shape) for i in op["index"]]
        val = op["value"]
        vals = [val] * len(idx) if isinstance(val, (int, float)) else list(val)
        if len(vals) != len(idx):
            raise ValueError("value/index length mismatch")
        for i, v in zip(idx, vals):
            if kind == "set":
                w[i] = v
            elif kind == "add":
                w[i] += v
            else:
                w[i] *= v
        return len(idx)
    if kind in ("scale_rows", "scale_cols"):
        if w.dim() != 2:
            raise ValueError(f"{kind} needs a 2-D tensor, got shape {tuple(w.shape)}")
        dim = 0 if kind == "scale_rows" else 1
        for i in op["index"]:
            if not 0 <= i < w.shape[dim]:
                raise IndexError(f"{kind} index {i} out of range for shape {tuple(w.shape)}")
            (w[i] if dim == 0 else w[:, i]).mul_(op["value"])
        return len(op["index"]) * w.shape[1 - dim]
    if kind == "lowrank_add":
        z = np.load(op["npz"])
        w += torch.from_numpy(z[op["u"]]).float() @ torch.from_numpy(z[op["v"]]).float().T
        return w.numel()
    raise ValueError(f"unknown op {kind!r}")


def _apply_tensor(orig: torch.Tensor, ops: list):
    """Apply ops to a bf16 tensor; return (new bf16 tensor, n_requested, n_changed)."""
    if orig.dtype != torch.bfloat16:
        raise TypeError(f"expected bfloat16 tensor, got {orig.dtype}")
    w = orig.float().clone()
    req = sum(_apply_op(w, op) for op in ops)
    new = w.to(torch.bfloat16)
    changed = int((new.view(torch.int16) != orig.view(torch.int16)).sum())
    return new, req, changed


def _shards(base_dir: Path):
    return sorted(Path(base_dir).glob("*.safetensors"))


def _by_tensor(edit):
    d = {}
    for op in edit["ops"]:
        d.setdefault(op["tensor"], []).append(op)
    return d


def _locate(base_dir: Path, names):
    """Map tensor name -> shard path; KeyError for unknown tensors."""
    where = {}
    for s in _shards(base_dir):
        with safe_open(str(s), "pt") as f:
            where.update({k: s for k in f.keys()})
    missing = [n for n in names if n not in where]
    if missing:
        raise KeyError(f"unknown tensor(s): {missing}")
    return where


def _clone(src: Path, dst: Path):
    src = src.resolve()
    if subprocess.run(["cp", "-c", str(src), str(dst)], capture_output=True).returncode:
        subprocess.run(["cp", str(src), str(dst)], check=True)


def apply_edit(edit: dict, base_dir: Path, out_dir: Path) -> EditStats:
    base_dir, out_dir = Path(base_dir), Path(out_dir)
    per = _by_tensor(edit)
    where = _locate(base_dir, per)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = EditStats(0, 0, sorted(per), {})
    touched = {where[n] for n in per}
    for p in base_dir.iterdir():
        if p not in touched:
            _clone(p, out_dir / p.name)
    for shard in touched:
        with safe_open(str(shard), "pt") as f:
            meta = f.metadata()
        tensors = load_file(str(shard.resolve()))
        for name in (n for n in per if where[n] == shard):
            new, req, ch = _apply_tensor(tensors[name], per[name])
            stats.delta_l2[name] = float((new.float() - tensors[name].float()).norm())
            stats.n_requested += req
            stats.n_changed += ch
            tensors[name] = new
        save_file(tensors, str(out_dir / shard.name), metadata=meta)
    return stats


def _place(w, mags, rng, tries=64, tol=0.25):
    """Pick a distinct random position + sign per magnitude so that the bf16-realized delta
    is nonzero and as close to the requested magnitude as bf16 spacing allows."""
    flat, used, out = w.flatten().float(), set(), []
    for m in mags:
        best = None
        for _ in range(8):
            pos = rng.integers(0, flat.numel(), size=tries)
            sg = rng.choice([-1.0, 1.0], size=tries)
            cur = flat[pos]
            real = ((cur + torch.from_numpy(sg).float() * float(m))
                    .to(torch.bfloat16).float() - cur).abs()
            for k in np.argsort(np.abs(real.numpy() / m - 1)):
                if real[k] > 0 and int(pos[k]) not in used:
                    err = abs(float(real[k]) / m - 1)
                    if best is None or err < best[0]:
                        best = (err, int(pos[k]), float(sg[k] * m))
                    break
            if best and best[0] <= tol:
                break
        if best is None:
            raise RuntimeError("could not place a delta that survives bf16 rounding")
        used.add(best[1])
        out.append((best[1], best[2]))
    return out


def random_control(edit: dict, base_dir: Path, seed: int, npz_dir: Path | None = None) -> dict:
    """Same tensors, same realized (post-bf16) delta magnitudes at random positions with random signs.
    The control dict's "realized" key records its actual n_changed / per-tensor delta_l2."""
    rng = np.random.default_rng(seed)
    base_dir = Path(base_dir)
    per = _by_tensor(edit)
    where = _locate(base_dir, per)
    ops, realized = [], {"n_changed": 0, "delta_l2": {}}
    for name, tops in per.items():
        with safe_open(str(where[name]), "pt") as f:
            orig = f.get_tensor(name)
        lr = [o for o in tops if o["op"] == "lowrank_add"]
        rest = [o for o in tops if o["op"] != "lowrank_add"]
        mine = []
        if rest:
            new, _, _ = _apply_tensor(orig, rest)
            d = (new.float() - orig.float()).flatten().numpy()
            placed = _place(orig, np.abs(d[d != 0]), rng)
            mine.append({"tensor": name, "op": "add",
                         "index": [list(map(int, np.unravel_index(p, orig.shape))) for p, _ in placed],
                         "value": [v for _, v in placed]})
        for k, o in enumerate(lr):
            out_dir = Path(npz_dir) if npz_dir else Path(o["npz"]).resolve().parent
            out_dir.mkdir(parents=True, exist_ok=True)
            z = np.load(o["npz"])
            u, v = z[o["u"]], z[o["v"]]
            ru, rv = rng.standard_normal(u.shape), rng.standard_normal(v.shape)
            ru *= np.linalg.norm(u) / np.linalg.norm(ru)
            rv *= np.linalg.norm(v) / np.linalg.norm(rv)
            path = str(out_dir / f"{edit['id']}_ctrl{seed}_{name.replace('/', '_')}_{k}.npz")
            np.savez(path, u=ru.astype(u.dtype), v=rv.astype(v.dtype))
            mine.append({"tensor": name, "op": "lowrank_add", "npz": path, "u": "u", "v": "v"})
        new, _, ch = _apply_tensor(orig, mine)
        realized["n_changed"] += ch
        realized["delta_l2"][name] = float((new.float() - orig.float()).norm())
        ops += mine
    return {"id": f"{edit['id']}-ctrl{seed}", "hypothesis": f"random control (seed {seed}) for {edit['id']}",
            "evidence": [], "ops": ops, "realized": realized}


def to_gguf(model_dir: Path, out: Path, outtype: str = "q8_0") -> str:
    out = Path(out)
    subprocess.run([sys.executable, str(REPO / "vendor/llama.cpp/convert_hf_to_gguf.py"), str(model_dir),
                    "--outtype", outtype, "--outfile", str(out)], check=True)
    h = hashlib.sha256()
    with open(out, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
