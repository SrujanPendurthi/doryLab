from scripts.paper.figures import make_all


def test_figures_written(tmp_path):
    make_all(tmp_path)
    for n in ("pareto", "headroom", "truncation", "spec"):
        f = tmp_path / f"fig_{n}.pdf"
        assert f.exists() and f.stat().st_size > 0
