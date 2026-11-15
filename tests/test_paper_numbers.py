import re

import pytest

from scripts.paper.numbers import compute, write


@pytest.fixture(scope="module")
def V():
    return {k: float(v) if re.fullmatch(r"[-+]?[\d.]+", v) else v for k, v in compute().items()}


def test_headline_numbers_match_decisions(V):
    for k, want in {"HoldoutPreFixedOneAcc": 0.600, "HoldoutPreFixedTwoAcc": 0.587, "HoldoutPreFixedFourAcc": 0.610,
                    "HoldoutPreOracleAcc": 0.670, "FinalPreFixedFourAcc": 0.630,
                    # post-regrade holdout (decisions.md "Task 8 (rerun after regrade)")
                    "HoldoutFixedOneAcc": 0.590, "HoldoutFixedTwoAcc": 0.587, "HoldoutFixedFourAcc": 0.607,
                    "HoldoutOracleAcc": 0.667, "FinalChosenAcc": 0.620, "FinalFloorAcc": 0.620,
                    "FinalFixedOneAcc": 0.590, "FinalFixedFourAcc": 0.610, "FinalOracleAcc": 0.670,
                    # decisions.md says n6 1.27; the geometric mean is 1.2646
                    "SpecNFourSpeedup": 1.21, "SpecNSixSpeedup": 1.265}.items():
        assert V[k] == pytest.approx(want, abs=0.0005 if V[k] < 1.1 else 0.01), k
    assert V["FinalFloorSec"] == pytest.approx(12.51, abs=0.01)
    assert V["FinalChosenSec"] == pytest.approx(20.28, abs=0.01)
    assert V["AneMs"] == pytest.approx(17.9, abs=0.1)
    assert V["MpsMs"] == pytest.approx(44.3, abs=0.1)
    assert V["SpecNFourAccept"] == pytest.approx(0.84, abs=0.01)
    assert V["OffloadSpeedup"] == pytest.approx(0.93, abs=0.01)
    assert V["NgramSpeedup"] == pytest.approx(0.74, abs=0.01)


def test_macro_names_valid(V):
    assert all(re.fullmatch(r"[A-Za-z]+", k) for k in V)


def test_write_outputs(tmp_path):
    write(tmp_path)
    for f in ("numbers.tex", "numbers.json", "tab_controllers_holdout.tex", "tab_controllers_final.tex",
              "tab_per_source.tex", "tab_systems.tex", "tab_truncation.tex", "tab_mmlu1024.tex"):
        assert (tmp_path / f).stat().st_size > 0


def test_text_rows_are_post_regrade(V):
    assert V["HoldoutTextAcc"] == pytest.approx(0.583, abs=0.0005)
    assert V["HoldoutTextDelta"] == pytest.approx(-0.003, abs=0.0005)
    assert (V["HoldoutTextLo"], V["HoldoutTextHi"]) == pytest.approx((-0.020, 0.013), abs=0.0005)
    assert V["FinalTextAcc"] == pytest.approx(0.630, abs=0.0005)
    assert V["FinalTextMeanK"] != V["FinalPreTextMeanK"]  # refit after regrade: tau 0.4326 -> 0.4312


def test_mmlu1024_cap_resweep(V):
    assert V["MmluCapFourAccBigCap"] == 60 and V["MmluCapFourTruncBigCap"] == 19
    assert (V["MmluCapFourWins"], V["MmluCapFourLosses"]) == (9, 5)
    assert V["MmluCapOneAccSmallCap"] == 48 and V["MmluCapTwoRescuedByCap"] == 13


def test_wsp_negative_result(V):
    # research/wsp/ledger.md: e001-e006 all rejected; e002 and e006 reached holdout
    assert (V["WspEdits"], V["WspScreenRejects"], V["WspHoldoutTested"]) == (6, 4, 2)
    assert (V["WspBestDacc"], V["WspBestDaccLo"], V["WspBestDaccHi"]) == pytest.approx((1.0, -2.3, 4.3))
    assert V["WspWorstDacc"] == pytest.approx(-0.7)
