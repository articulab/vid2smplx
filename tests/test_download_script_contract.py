"""download_models.sh parses the MODELS table out of checks.py with ast.

That is a real coupling between a shell script and a Python module, and nothing else
tests it: the script only runs during a full install, so a change to MODELS that breaks
the parse fails at Phase 5 on a fresh machine and nowhere else. It has happened --
replacing the literal note "auto" with the DOWNLOADABLE constant turned every row into
a non-literal and `ast.literal_eval(tuple)` raised, aborting the install before any
weight was fetched.
"""
import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHECKS = REPO / "vid2smplx" / "checks.py"
SCRIPT = REPO / "scripts" / "download_models.sh"


def _models_node():
    mod = ast.parse(CHECKS.read_text())
    for n in mod.body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "MODELS":
            return n.value
    raise AssertionError("checks.py no longer defines a module-level MODELS")


def test_every_models_row_exposes_a_literal_path_and_floor():
    """Fields 1 and 3 are what the downloader reads; they must stay literals."""
    rows = _models_node().elts
    assert rows, "MODELS is empty"
    for e in rows:
        path = ast.literal_eval(e.elts[1])       # raises if someone makes it a constant
        floor = ast.literal_eval(e.elts[3])
        assert isinstance(path, str) and path
        assert isinstance(floor, int) and floor > 0, f"{path}: min_bytes must be a positive int"


def test_the_downloader_still_reads_the_fields_it_thinks_it_does():
    """If the script's indices drift from the table's shape, catch it here, not on a user's machine."""
    src = SCRIPT.read_text()
    assert "MODELS" in src, "download_models.sh no longer reads the MODELS table"
    idx = sorted(set(re.findall(r"elts\[(\d+)\]", src)))
    assert idx == ["1", "3"], f"downloader now reads fields {idx}; update this test and the table contract"
    width = {len(e.elts) for e in _models_node().elts}
    assert width == {4}, f"MODELS rows are {width}-wide; the downloader indexes 1 and 3"
