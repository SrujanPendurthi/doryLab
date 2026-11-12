from pathlib import Path

import pytest

from dory.server import server_args


def test_server_args_draft():
    a = server_args(Path("t.gguf"), 4, 8080, slots=1, draft=Path("m.gguf"))
    assert a[a.index("-md") + 1] == "m.gguf"
    assert "--override-kv" not in a  # llama-server copies target kv_overrides to the draft
    assert a[a.index("--spec-draft-n-max") + 1] == "4"
    assert a[a.index("--spec-type") + 1] == "draft-simple"  # default is none: draft loads but never drafts


def test_server_args_draft_only_k4():
    with pytest.raises(ValueError):
        server_args(Path("t.gguf"), 2, 8080, draft=Path("m.gguf"))
