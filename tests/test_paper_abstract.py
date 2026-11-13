from pathlib import Path

ABSTRACT = Path(__file__).resolve().parents[1] / "paper" / "abstract.txt"


def test_metadata_abstract_length_and_plain_text():
    a = ABSTRACT.read_text().strip()
    assert 0 < len(a) <= 1920
    assert not any(c in a for c in "\\${}")
