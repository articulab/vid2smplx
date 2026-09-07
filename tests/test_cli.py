"""Smoke tests that need no GPU, no conda env, no weights.  Run: python -m pytest tests/  (or python tests/test_cli.py)"""
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest  # noqa: E402

from vid2smplx.cli import build_parser, validate_run_args  # noqa: E402
from vid2smplx.checks import LINKS, MODELS, doctor, make_links, resolve  # noqa: E402


def test_flags_and_aliases():
    p = build_parser()
    a = p.parse_args(["run", "v.mp4", "--final_incam", "--batch-size", "8", "--no-face"])
    assert a.final_incam and a.batch_size == 8 and a.no_face
    b = p.parse_args(["run", "v.mp4", "--final-incam", "--output_dir", "x"])
    assert b.final_incam and b.output_dir == "x"
    assert p.parse_args(["doctor"]).cmd == "doctor"
    assert p.parse_args(["render", "out/clip", "--layers", "final"]).layers == "final"


def test_doctor_reports_missing(tmp_path, capsys):
    assert doctor(repo=tmp_path, check_env=False) is False
    out = capsys.readouterr().out
    assert "[MISS] SMPL-X: models/smplx/SMPLX_NEUTRAL.npz" in out
    assert "smpl-x.is.tue.mpg.de" in out


def _touch_all(tmp_path, skip=()):
    for group, rel, _ in MODELS:
        if group in skip:
            continue
        p = resolve(rel, tmp_path, home=tmp_path / "home")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("")


def test_doctor_passes_when_everything_exists(tmp_path, capsys):
    _touch_all(tmp_path)
    made = make_links(tmp_path)
    assert sorted(made) == sorted(l for l, _ in LINKS)
    assert make_links(tmp_path) == []  # idempotent
    assert (tmp_path / LINKS[0][0]).resolve() == (tmp_path / LINKS[0][1]).resolve()
    assert doctor(repo=tmp_path, check_env=False, home=tmp_path / "home") is True
    assert "[MISS]" not in capsys.readouterr().out


def test_empty_placeholder_dir_is_replaced_by_link(tmp_path, capsys):
    """hamer_demo_data.tar.gz ships an empty _DATA/data/mano/; it must not shadow the link."""
    _touch_all(tmp_path)
    link, target = LINKS[1]                      # hamer/_DATA/data/mano
    (tmp_path / link).mkdir(parents=True)        # the empty placeholder
    assert link in make_links(tmp_path)
    assert (tmp_path / link).resolve() == (tmp_path / target).resolve()
    assert doctor(repo=tmp_path, check_env=False, home=tmp_path / "home") is True

    # and an empty placeholder must never read as [OK]
    (tmp_path / link).unlink()
    (tmp_path / link).mkdir()
    assert doctor(repo=tmp_path, check_env=False, home=tmp_path / "home") is False
    assert "[MISS] link" in capsys.readouterr().out

    # a dangling symlink must be repairable, not a permanent MISS
    (tmp_path / link).rmdir()
    (tmp_path / link).symlink_to(tmp_path / "nowhere")
    assert doctor(repo=tmp_path, check_env=False, home=tmp_path / "home") is False
    assert link in make_links(tmp_path)
    assert doctor(repo=tmp_path, check_env=False, home=tmp_path / "home") is True


def test_doctor_skip_groups(tmp_path):
    _touch_all(tmp_path, skip=("EMICA", "Gaze"))
    make_links(tmp_path)
    h = tmp_path / "home"
    assert doctor(repo=tmp_path, check_env=False, skip={"EMICA", "Gaze"}, home=h) is True
    assert doctor(repo=tmp_path, check_env=False, home=h) is False


def test_docs_list_every_model():
    docs = (Path(__file__).resolve().parents[1] / "docs" / "models.md").read_text()
    for _, rel, _ in MODELS:
        p = Path(rel)
        assert p.name in docs or p.parent.name in docs, f"{rel} missing from docs/models.md"


def test_blackwell_pins_match_frozen_requirements():
    """Script pins must match the frozen requirements; drift built a broken env."""
    root = Path(__file__).resolve().parents[1] / "specific_installation"
    frozen = {}
    for line in (root / "requirements_blackwell.txt").read_text().splitlines():
        if "==" in line and not line.lstrip().startswith("#"):
            name, _, ver = line.strip().partition("==")
            frozen[name.strip().lower().replace("_", "-")] = ver.split()[0].strip()

    script = (root / "install_blackwell.sh").read_text()
    mismatched = []
    for name, ver in frozen.items():
        for m in re.finditer(rf'(?<![\w.-]){re.escape(name)}==([\w.+]+)', script, re.I):
            if m.group(1) != ver:
                mismatched.append(f"{name}: script pins {m.group(1)}, frozen says {ver}")
    assert not mismatched, "install_blackwell.sh disagrees with requirements_blackwell.txt:\n  " \
                           + "\n  ".join(mismatched)


if __name__ == "__main__":  # ponytail: pytest optional (pytest.raises needs pytest anyway)
    import io, contextlib, tempfile
    class _Cap:
        def readouterr(self):
            return type("O", (), {"out": buf.getvalue()})()
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            buf = io.StringIO()
            with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(buf):
                kw = {}
                if "tmp_path" in fn.__code__.co_varnames: kw["tmp_path"] = Path(d)
                if "capsys" in fn.__code__.co_varnames: kw["capsys"] = _Cap()
                fn(**kw)
            print("ok", name)


def test_rejects_bad_values():
    p = build_parser()
    for argv in (["run", "v.mp4", "--percent", "0"], ["run", "v.mp4", "--percent", "101"],
                 ["run", "v.mp4", "--downsample", "0"], ["run", "v.mp4", "--batch-size", "-1"],
                 ["run", "v.mp4", "--face-method", "foo"], ["run"], ["render"]):
        with pytest.raises(SystemExit):
            p.parse_args(argv)


def test_deprecated_flags_are_noops(tmp_path, capsys):
    vid = tmp_path / "v.mp4"; vid.write_bytes(b"")
    argv = ["run", str(vid), "--production", "--skip_render", "--use_gvhmr_focal", "--final-incam", "--full-debug"]
    a = build_parser().parse_args(argv)
    validate_run_args(a, argv, error=pytest.fail)
    out = capsys.readouterr().out
    assert out.count("[DEPRECATED]") == 3
    assert a.full_debug and not a.final_incam           # full-debug wins, final-incam dropped with a note
    assert "ignoring --final-incam" in out


def test_missing_video_is_an_error(tmp_path):
    argv = ["run", str(tmp_path / "nope.mp4")]
    a = build_parser().parse_args(argv)
    errs = []
    validate_run_args(a, argv, error=errs.append)
    assert errs and "video not found" in errs[0]
