import json

from dory.bench import load_speed, summarize
from dory.client import Completion


def _c(n_gen, gen_tps):
    return Completion("", 1000, n_gen, 500.0, gen_tps, 1.0)


def test_summarize():
    s = summarize([_c(100, 10.0), _c(300, 30.0)], 20.0)
    assert s["agg_tps"] == 20.0 and s["gen_tps_per_req"] == 20.0 and s["prompt_tps"] == 500.0


def test_load_speed_int_keys(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"1": {"prompt_tps": 1.0, "gen_tps": 2.0}, "4": {"prompt_tps": 3.0, "gen_tps": 4.0}}))
    assert load_speed(p) == {1: {"prompt_tps": 1.0, "gen_tps": 2.0}, 4: {"prompt_tps": 3.0, "gen_tps": 4.0}}
