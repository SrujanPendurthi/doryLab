import pytest
import torch
from safetensors.torch import load_file, save_file

from dory.wsp.edits import apply_edit, random_control


@pytest.fixture
def base(tmp_path):
    d = tmp_path / "base"
    d.mkdir()
    save_file({"a": torch.ones(8, 8, dtype=torch.bfloat16),
               "b": torch.full((8, 8), 2.0, dtype=torch.bfloat16)}, str(d / "model.safetensors"))
    (d / "config.json").write_text("{}")
    return d


def _bytes(t):
    return t.view(torch.int16)


def test_apply_set_add_scale(base, tmp_path):
    e = {"id": "x", "hypothesis": "", "evidence": [], "ops": [
        {"tensor": "a", "op": "set", "index": [[0, 0]], "value": [5.0]},
        {"tensor": "a", "op": "add", "index": [[1, 1]], "value": [0.5]},
        {"tensor": "a", "op": "scale", "index": [[2, 2]], "value": 4.0}]}
    out = tmp_path / "out"
    st = apply_edit(e, base, out)
    a = load_file(str(out / "model.safetensors"))["a"].float()
    assert (a[0, 0], a[1, 1], a[2, 2]) == (5.0, 1.5, 4.0)
    mask = torch.ones(8, 8, dtype=torch.bool)
    for i in range(3):
        mask[i, i] = False
    assert (a[mask] == 1.0).all()
    assert st.n_requested == st.n_changed == 3 and (out / "config.json").exists()


def test_unknown_tensor_or_index_raises(base, tmp_path):
    with pytest.raises(KeyError):
        apply_edit({"id": "x", "ops": [{"tensor": "zz", "op": "set", "index": [[0, 0]], "value": [1.0]}]}, base, tmp_path / "o1")
    with pytest.raises(IndexError):
        apply_edit({"id": "x", "ops": [{"tensor": "a", "op": "set", "index": [[8, 0]], "value": [1.0]}]}, base, tmp_path / "o2")


def test_n_changed_counts_bf16_noop(base, tmp_path):
    st = apply_edit({"id": "x", "ops": [{"tensor": "a", "op": "add", "index": [[0, 0]], "value": [1e-6]}]}, base, tmp_path / "o")
    assert (st.n_requested, st.n_changed) == (1, 0)


def test_control_matches_size(base, tmp_path):
    e = {"id": "x", "ops": [{"tensor": "a", "op": "add", "index": [[0, 0], [1, 1], [2, 2]], "value": [0.5, -0.25, 1.0]}]}
    c = random_control(e, base, seed=0)
    assert {o["tensor"] for o in c["ops"]} == {"a"}
    se, sc = apply_edit(e, base, tmp_path / "e"), apply_edit(c, base, tmp_path / "c")
    assert se.n_changed == sc.n_changed == 3
    ae = load_file(str(tmp_path / "e/model.safetensors"))["a"].float() - 1
    ac = load_file(str(tmp_path / "c/model.safetensors"))["a"].float() - 1
    assert torch.equal(ae[ae != 0].abs().sort().values, ac[ac != 0].abs().sort().values)
    assert not torch.equal(ae != 0, ac != 0)


def test_untouched_tensor_identical(base, tmp_path):
    out = tmp_path / "out"
    apply_edit({"id": "x", "ops": [{"tensor": "a", "op": "set", "index": [[0, 0]], "value": [9.0]}]}, base, out)
    orig, new = load_file(str(base / "model.safetensors")), load_file(str(out / "model.safetensors"))
    assert torch.equal(_bytes(orig["b"]), _bytes(new["b"]))
