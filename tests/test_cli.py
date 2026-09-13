"""Smoke tests that need no GPU, no conda env, no weights.  Run: python -m pytest tests/  (or python tests/test_cli.py)"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
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
    assert doctor(repo=tmp_path, check_env=False, check_patches=False) is False
    out = capsys.readouterr().out
    assert "[MISS] SMPL-X: models/smplx/SMPLX_NEUTRAL.npz" in out
    assert "smpl-x.is.tue.mpg.de" in out


def _touch_all(tmp_path, skip=()):
    """Create every model at its declared minimum size — doctor checks bytes, not existence.

    Sparse (truncate), because the real checkpoints total ~13 GB.
    """
    def _sized(p: Path, n: int):
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            f.truncate(n)

    for group, rel, _, min_bytes in MODELS:
        if group in skip:
            continue
        p = resolve(rel, tmp_path, home=tmp_path / "home")
        if rel.endswith("/models"):                      # EMICA asset dir, not a file
            p.mkdir(parents=True, exist_ok=True)
            _sized(p / "w.bin", min_bytes)
        else:
            _sized(p, min_bytes)


def test_doctor_passes_when_everything_exists(tmp_path, capsys):
    _touch_all(tmp_path)
    made = make_links(tmp_path)
    assert sorted(made) == sorted(l for _, l, _ in LINKS)
    assert make_links(tmp_path) == []  # idempotent
    assert (tmp_path / LINKS[0][1]).resolve() == (tmp_path / LINKS[0][2]).resolve()
    assert doctor(repo=tmp_path, check_env=False, check_patches=False, home=tmp_path / "home") is True
    assert "[MISS]" not in capsys.readouterr().out


def test_doctor_rejects_zero_byte_weight(tmp_path, capsys):
    """A truncated download exists but is unusable; doctor must not call it OK."""
    _touch_all(tmp_path)
    make_links(tmp_path)
    (tmp_path / "models/L2CSNet_gaze360.pkl").write_bytes(b"")
    assert doctor(repo=tmp_path, check_env=False, check_patches=False, home=tmp_path / "home") is False
    out = capsys.readouterr().out
    assert "[CORRUPT] Gaze: models/L2CSNet_gaze360.pkl" in out
    assert "vid2smplx download" in out


def test_doctor_skip_groups(tmp_path, capsys):
    """--no-face must not be blocked by EMICA assets, including their symlinks."""
    _touch_all(tmp_path, skip=("EMICA", "Gaze"))
    make_links(tmp_path)
    for _, link, _ in LINKS:                             # EMICA links dangle with no assets
        assert (tmp_path / link).is_symlink()
    h = tmp_path / "home"
    assert doctor(repo=tmp_path, check_env=False, check_patches=False, skip={"EMICA", "Gaze"}, home=h) is True
    assert "inferno/assets" not in capsys.readouterr().out
    assert doctor(repo=tmp_path, check_env=False, check_patches=False, home=h) is False


def test_docs_list_every_model():
    docs = (Path(__file__).resolve().parents[1] / "docs" / "models.md").read_text()
    for _, rel, _, _ in MODELS:
        p = Path(rel)
        assert p.name in docs or p.parent.name in docs, f"{rel} missing from docs/models.md"


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


def test_deprecated_flags_are_noops(capsys):
    vid = REPO / "examples" / "clip_talking.mp4"   # a real video: validation now probes it
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


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="needs ffprobe")
def test_non_video_is_rejected_before_any_model_loads(tmp_path):
    """A text file named .mp4 used to reach Step 1/5 and die in a traceback."""
    fake = tmp_path / "notavideo.mp4"; fake.write_text("hello\n")
    argv = ["run", str(fake)]
    a = build_parser().parse_args(argv)
    errs = []
    validate_run_args(a, argv, error=errs.append)
    assert errs and "no video stream" in errs[0] and "ffprobe" in errs[0]


# ---- provenance stamps (BLOCKER-1): a changed input must never be reused ----

def _vid(tmp_path, name="A.mp4", data=b"x" * 100):
    p = tmp_path / name
    p.write_bytes(data)
    return p


def test_stamp_reuses_only_the_same_input_and_flags(tmp_path):
    from vid2smplx.cli import stamp_ok, stamp_write, video_identity
    src = _vid(tmp_path)
    art = tmp_path / "out.pt"; art.write_text("result")
    key = {"video": video_identity(src), "static_cam": True}

    ok, why = stamp_ok(art, key)                              # unstamped -> never reused
    assert ok is False and "no provenance stamp" in why       # wording is free to change
    stamp_write(art, key)
    assert stamp_ok(art, key)[0] is True                      # unchanged -> reuse

    src.write_bytes(b"y" * 250)                               # same NAME, different content
    changed = {"video": video_identity(src), "static_cam": True}
    ok, why = stamp_ok(art, changed)
    assert ok is False and "changed" in why and "video" in why

    ok, why = stamp_ok(art, {**key, "static_cam": False})      # a flag that shapes the stage
    assert ok is False and "static_cam" in why


def test_stamp_invalidates_when_the_code_changes(tmp_path, monkeypatch):
    """A `git pull` that changes code or bumps a submodule must not be answered with [SKIP].

    Before this, stamp keys held inputs and flags only, so a re-run into an existing
    --output-dir printed "[SKIP] Already exists" and wrote SUCCESS over output from the
    old code.
    """
    from vid2smplx import cli, provenance
    art = tmp_path / "out.pt"; art.write_text("result")
    key = {"video": {"name": "a.mp4", "size": 1, "mtime_ns": 2}}

    monkeypatch.setattr(provenance, "code_identity",
                        lambda repo=None: (("git_commit", "aaaa"), ("git_dirty", False)))
    cli.stamp_write(art, key)
    assert cli.stamp_ok(art, key)[0] is True

    monkeypatch.setattr(provenance, "code_identity",       # submodule pin bumped by the pull
                        lambda repo=None: (("git_commit", "aaaa"), ("git_dirty", False),
                                           ("git_submodules", "beef GVHMR")))
    ok, why = cli.stamp_ok(art, key)
    assert ok is False and "checkout changed" in why

    monkeypatch.setattr(provenance, "code_identity",       # repo HEAD moved
                        lambda repo=None: (("git_commit", "bbbb"), ("git_dirty", False)))
    assert cli.stamp_ok(art, key)[0] is False


def test_stamped_key_records_the_code_identity(tmp_path):
    """Not just "it re-runs": the stamp on disk must actually carry the code fields."""
    import json
    from vid2smplx.cli import stamp_path, stamp_write
    art = tmp_path / "out.pt"; art.write_text("result")
    stamp_write(art, {"video": None})
    code = json.loads(stamp_path(art).read_text())["key"]["code"]
    assert set(code) == {"git_commit", "git_dirty", "git_submodules"}


def test_stale_artifact_is_deleted_not_reused(tmp_path):
    """reuse_or_clear must remove the stale artifact: a half-old output dir is worse than none."""
    from vid2smplx.cli import reuse_or_clear, stamp_path, stamp_write
    art = tmp_path / "r.pt"; art.write_text("old")
    side = tmp_path / "stage"; side.mkdir(); (side / "junk.bin").write_text("old")
    stamp_write(art, {"v": 1})
    assert reuse_or_clear(art, {"v": 1}, "stage", extra_clear=[side]) is True
    assert art.exists() and side.exists()
    assert reuse_or_clear(art, {"v": 2}, "stage", extra_clear=[side]) is False
    assert not art.exists() and not stamp_path(art).exists() and not side.exists()



# ---- a killed run must never leave an artifact a re-run trusts ----

def test_stamp_does_not_bless_a_truncated_npz(tmp_path):
    """The stamp proves provenance, not that the bytes are whole.

    IK rewrites smplx_params.npz in place, under the merge stamp. A kill mid-write used
    to leave a 0-byte npz the next run happily SKIPped merge for.
    """
    import numpy as np
    from vid2smplx.cli import reuse_or_clear, stamp_path, stamp_write
    art = tmp_path / "smplx_params.npz"
    np.savez(art, body_pose=np.zeros((10, 63), np.float32))
    stamp_write(art, {"v": 1})
    assert reuse_or_clear(art, {"v": 1}, "merge") is True          # intact -> reuse

    raw = art.read_bytes(); art.write_bytes(raw[:len(raw) // 2])   # killed mid-write
    assert reuse_or_clear(art, {"v": 1}, "merge") is False         # must NOT be reused
    assert not art.exists() and not stamp_path(art).exists()       # and must be cleared


def test_sigkill_mid_write_leaves_the_previous_artifact_intact(tmp_path):
    """save_npz_atomic: the final path only ever holds a whole file."""
    import os, signal, subprocess, sys, time
    import numpy as np
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    art = tmp_path / "smplx_params.npz"
    np.savez(art, body_pose=np.zeros((300000, 63), np.float32))
    before = art.read_bytes()

    writer = tmp_path / "w.py"
    writer.write_text(
        "import sys; sys.path.insert(0, %r)\n"
        "import numpy as np\n"
        "from utils import save_npz_atomic\n"
        "save_npz_atomic(sys.argv[1], compressed=False,\n"
        "                body_pose=np.ones((300000, 63), np.float32))\n" % scripts)
    proc = subprocess.Popen([sys.executable, str(writer), str(art)])
    killed = False
    for _ in range(20000):
        time.sleep(0.0005)
        if any(f.name.startswith(".smplx_params.npz.") for f in tmp_path.iterdir()):
            os.kill(proc.pid, signal.SIGKILL); killed = True; break
        if proc.poll() is not None:
            break
    proc.wait()
    if not killed:
        pytest.skip("write finished before the kill landed")
    assert art.read_bytes() == before          # untouched, not truncated
    np.load(art)["body_pose"]                  # and still loadable


# ---- actionable child errors (MAJOR-4) ----

@pytest.mark.parametrize("stderr, expect", [
    ("torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB", "--batch-size 16"),
    ('File "hmr4d/utils/preproc/tracker.py", line 81\nIndexError: list index out of range', "No person was detected"),
    ("av.error.InvalidDataError: Invalid NAL unit size (29960 > 27268)", "truncated or corrupt"),
    ("died with <Signals.SIGXFSZ: 25>", "ulimit -f"),
    ("OSError: [Errno 28] No space left on device", "disk"),
    ("RuntimeError: --person 5 was requested but only 2 person track(s) were found", "--person 5"),
])
def test_child_failures_map_to_one_actionable_line(stderr, expect):
    import subprocess as sp
    from vid2smplx.cli import explain_child_failure
    e = sp.CalledProcessError(1, ["python", "demo.py"], output="", stderr=stderr)
    msg = explain_child_failure(e)
    assert expect in msg, msg
    assert "Traceback" not in msg


def test_unrecognised_failure_still_says_what_to_do():
    import subprocess as sp
    from vid2smplx.cli import explain_child_failure
    msg = explain_child_failure(sp.CalledProcessError(3, ["python", "run_emica.py"], output="", stderr="weird"))
    assert "run_emica.py" in msg and "3" in msg


# ---- multi-person (BLOCKER-3) ----

def test_multi_person_is_never_silent():
    from vid2smplx.cli import multi_person_failure
    one = {"n_tracks": 1, "chosen_rank": 0, "tracks": [{"rank": 0, "track_id": 1, "n_frames": 9,
                                                        "bbx_xyxy_median": [0, 0, 10, 10]}]}
    assert multi_person_failure(one, None) == ""
    two = {"n_tracks": 2, "chosen_rank": 0, "tracks": [
        {"rank": 0, "track_id": 1, "n_frames": 100, "bbx_xyxy_median": [0, 60, 479, 842]},
        {"rank": 1, "track_id": 2, "n_frames": 100, "bbx_xyxy_median": [600, 248, 900, 646]}]}
    msg = multi_person_failure(two, None)
    assert "2 people" in msg and "rank 0 <- CHOSEN" in msg and "--person 0" in msg
    assert "600" in msg                                  # the discarded person is named, not dropped
    assert "--person 1" in multi_person_failure(two, 1)


def _summary(*extra, total=1800):
    """chosen track spanning the clip, plus the given (n_frames, overlap, area_median) others."""
    tracks = [{"rank": 0, "track_id": 1, "n_frames": total, "n_overlap_frames": total,
               "area_median": 0.2, "area_share": 1.0, "bbx_xyxy_median": [0, 60, 479, 842]}]
    for i, (n, ov, area) in enumerate(extra, start=1):
        tracks.append({"rank": i, "track_id": i + 1, "n_frames": n, "n_overlap_frames": ov,
                       "area_median": area, "area_share": round(area / 0.2, 4),
                       "bbx_xyxy_median": [600, 248, 900, 646]})
    return {"n_tracks": len(tracks), "n_frames_total": total, "chosen_rank": 0,
            "chosen_track_id": 1, "tracks": tracks}


def test_dyad_still_fails_hard():
    """The invariant: two co-present, similarly sized subjects must never be a silent SUCCESS."""
    from vid2smplx.cli import multi_person_failure, spurious_track_warning
    s = _summary((1800, 1800, 0.09))                     # measured on a real two-person clip
    assert "2 people" in multi_person_failure(s, None)
    assert spurious_track_warning(s) == ""


def test_three_frame_blip_warns_but_does_not_fail():
    """A 3-frame passer-by must not kill a 96-clip SLURM array."""
    from vid2smplx.cli import multi_person_failure, spurious_track_warning
    s = _summary((3, 3, 0.09))
    assert multi_person_failure(s, None) == ""
    w = spurious_track_warning(s)
    assert "track id 2" in w and "3 frames" in w         # named, with its extent


def test_yolo_id_switch_is_not_a_second_person():
    """A re-identified SAME person occupies disjoint frames: long, full-size, zero overlap."""
    from vid2smplx.cli import multi_person_failure, spurious_track_warning
    s = _summary((900, 0, 0.2))
    assert multi_person_failure(s, None) == ""
    assert "0 co-present" in spurious_track_warning(s)


def test_small_distant_track_is_not_a_second_person():
    from vid2smplx.cli import multi_person_failure
    assert multi_person_failure(_summary((1800, 1800, 0.002)), None) == ""  # 0.2% of frame: too small
    # A real second subject sitting further back: only 0.24x the chosen bbox, still fatal.
    assert multi_person_failure(_summary((1800, 1800, 0.048)), None) != ""


def test_unmeasured_track_counts_as_a_person():
    """Old bbx.pt without overlap/area must escalate, never downgrade itself to a warning."""
    from vid2smplx.cli import multi_person_failure
    legacy = {"n_tracks": 2, "chosen_rank": 0, "tracks": [
        {"rank": 0, "track_id": 1, "n_frames": 100, "bbx_xyxy_median": [0, 60, 479, 842]},
        {"rank": 1, "track_id": 2, "n_frames": 3, "bbx_xyxy_median": [600, 248, 900, 646]}]}
    assert multi_person_failure(legacy, None) != ""


# ---- submodule changes (BLOCKER-1: must survive a fresh clone) ----

def test_required_submodule_changes_are_present_in_the_live_tree():
    """The checked-out GVHMR must carry every change vid2smplx needs, from the pin alone."""
    from vid2smplx import setup_submodules as S
    assert all(r[0] == "OK" for r in S.patch_status()), S.patch_status()
    assert S.main() == 0


def test_missing_change_is_detected_and_reported_actionably(tmp_path):
    """A submodule at the wrong commit must fail loudly, naming the pin and the fix."""
    from vid2smplx import setup_submodules as S
    sub, rel, name, marker = S.REQUIRED_MARKERS[0]
    p = tmp_path / sub / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("def get_one_track(self, video_path):\n    pass\n")   # upstream, marker absent
    with pytest.raises(S.PatchError) as e:
        S.verify_marker(sub, rel, name, marker, repo=tmp_path)
    msg = str(e.value)
    assert "WRONG COMMIT" in msg and rel in msg and "submodule update --init GVHMR" in msg
    assert "stale" in msg
    assert S.patch_status(tmp_path)[0][0] == "MISS"


def test_verification_never_writes_to_the_submodule(tmp_path):
    """Verify-only: the old apply path is gone, so nothing under the submodule may change."""
    from vid2smplx import setup_submodules as S
    sub, rel, name, marker = S.REQUIRED_MARKERS[0]
    p = tmp_path / sub / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("upstream body, no marker\n")
    S.patch_status(tmp_path)
    assert p.read_text() == "upstream body, no marker\n"
    assert not hasattr(S, "apply_patch") and not hasattr(S, "PATCHES")


# ---- render hardening (MAJOR-5) ----

def test_render_rejects_unknown_layers():
    from vid2smplx import VALID_LAYERS
    p = build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["render", "d", "--layers", "bogus"])
    with pytest.raises(SystemExit):
        p.parse_args(["render", "d", "--layers", "final,bogus"])
    assert p.parse_args(["render", "d", "--layers", "final,global"]).layers == "final,global"
    for layer in VALID_LAYERS:                            # every documented layer is accepted
        assert p.parse_args(["render", "d", "--layers", layer]).layers == layer


def test_render_validates_clip_dir(tmp_path):
    from vid2smplx.cli import validate_render_args
    a = build_parser().parse_args(["render", str(tmp_path / "nope")])
    errs = []; validate_render_args(a, errs.append)
    assert errs and "clip dir not found" in errs[0]

    f = tmp_path / "a.mp4"; f.write_text("x")
    a = build_parser().parse_args(["render", str(f)])
    errs = []; validate_render_args(a, errs.append)
    assert errs and "not a clip directory" in errs[0]

    d = tmp_path / "empty"; d.mkdir()
    a = build_parser().parse_args(["render", str(d)])
    errs = []; validate_render_args(a, errs.append)
    assert errs and "not a vid2smplx output dir" in errs[0]

    # smplx_params.npz alone is what `run --cleanup` leaves: accepted here, it used to die
    # deep inside render.py with "Video not found". Now it is rejected with the remedy.
    (d / "smplx_params.npz").write_text("x")
    a = build_parser().parse_args(["render", str(d)])
    errs = []; validate_render_args(a, errs.append)
    assert errs and "--video" in errs[0]

    (d / "gvhmr" / "empty").mkdir(parents=True)
    (d / "gvhmr" / "empty" / "0_input_video.mp4").write_bytes(b"x")
    (d / "gvhmr" / "empty" / "hmr4d_results.pt").write_bytes(b"x")
    a = build_parser().parse_args(["render", str(d)])
    errs = []; validate_render_args(a, errs.append)
    assert not errs


def test_render_layers_match_the_renderer():
    """cli choices and render.py must never drift apart."""
    from vid2smplx import VALID_LAYERS
    src = (REPO / "scripts" / "render.py").read_text()
    assert "from vid2smplx import VALID_LAYERS" in src
    assert set(VALID_LAYERS) == {"gvhmr", "hands", "face", "final", "global"}


# ---- VRAM display / threshold (MINOR-8) ----

@pytest.mark.parametrize("mib, shown, warned", [
    (8188, 8, False),    # a real "8 GB" card: the documented minimum, must NOT warn
    (8192, 8, False),
    (6141, 6, True),     # RTX 4050 6 GB
    (4096, 4, True),
])
def test_vram_row_uses_rounded_gib(mib, shown, warned):
    from vid2smplx.checks import _vram_row
    total = mib * 2 ** 20
    assert round(total / 2 ** 30) == shown
    row = _vram_row(("OK", "cuda", f"Some GPU ({shown} GB)"))
    assert (row[0] == "WARN") is warned, row


# ---- output dir lock (MINOR-9) ----

def test_dir_lock_refuses_a_second_run(tmp_path):
    from vid2smplx.cli import dir_lock
    with dir_lock(tmp_path / "out"):
        with pytest.raises(SystemExit) as ei:
            with dir_lock(tmp_path / "out"):
                pass
        assert "already running" in str(ei.value)
    with dir_lock(tmp_path / "out"):      # released, so it works again
        pass


# ---- VRAM preflight: recommend a card, never pretend to predict a peak ----

def test_vram_warning_is_quiet_on_every_card_length_pair_that_actually_ran():
    """Every measured success in docs/benchmarks.md must come out silent.

    A preflight that warns on hardware the docs call supported teaches people to ignore it,
    and across 158 corpus run logs there is not one OOM line to point at.
    """
    from vid2smplx.checks import vram_warning
    assert vram_warning(38, 8.0) == ""              # documented smoke test, 8 GB card
    assert vram_warning(369, 8.0) == ""             # where the 5.1 GB was measured
    assert vram_warning(370, 8.0) == ""             # one frame more is not a different machine
    assert vram_warning(12_520, 45.4) == ""         # rtx8000, 12.5k frames -> rc=0
    assert vram_warning(35_755, 45.4) == ""         # rtx8000, 20-min video -> rc=0
    assert vram_warning(35_755, 80.0) == ""         # h100, 20-min video    -> rc=0


def test_vram_warning_fires_only_past_the_measured_range_and_says_what_to_do():
    """The one case docs do not cover: a long clip on a card below the ~16 GB recommendation."""
    from vid2smplx.checks import vram_warning, LONG_VIDEO_VRAM_GB, MEASURED_TO_FRAMES
    assert vram_warning(MEASURED_TO_FRAMES, 8.0) == "", "warned inside the measured range"
    warn = vram_warning(MEASURED_TO_FRAMES + 1, 8.0)
    assert warn, "a 20-min clip on an 8 GB card got no warning at all"
    assert f"~{LONG_VIDEO_VRAM_GB} GB" in warn      # a real card size, and the documented one
    assert "cut the video" in warn                  # what to do instead
    assert "cannot predict" in warn                 # honest about why there is no number
    # The recommended card clears it; so does anything bigger.
    assert vram_warning(MEASURED_TO_FRAMES + 1, float(LONG_VIDEO_VRAM_GB)) == ""
    assert vram_warning(200_000, 45.4) == ""


def test_vram_recommendation_is_a_card_size_the_docs_state():
    """README.md and docs/install.md quote these two numbers; they must come from here."""
    from vid2smplx.checks import recommended_vram_gb, MIN_VRAM_GB, LONG_VIDEO_VRAM_GB
    assert (MIN_VRAM_GB, LONG_VIDEO_VRAM_GB) == (8, 16)
    assert recommended_vram_gb(369) == MIN_VRAM_GB
    assert recommended_vram_gb(35_755) == LONG_VIDEO_VRAM_GB
    for doc in ("README.md", "docs/install.md"):
        text = (REPO / doc).read_text()
        assert f"{MIN_VRAM_GB} GB" in text and f"{LONG_VIDEO_VRAM_GB} GB" in text, doc


def test_vram_warning_is_silent_without_data():
    from vid2smplx.checks import vram_warning
    assert vram_warning(0, 45.4) == ""              # unknown frame count -> no guess
    assert vram_warning(35755, 0.0) == ""           # no GPU visible      -> no guess


# ---- --no-hands must mean no hands, not "whatever the last run left behind" ----

def _seed_finished_run(tmp_path, monkeypatch, *, with_hands=True):
    """A completed run on disk, plus stubs for everything that needs a GPU or ffmpeg.

    Every stage artifact is stamped with the key the code will recompute, so each
    reuse_or_clear returns True and no real child process is ever needed.
    """
    import numpy as np
    from vid2smplx import cli

    video = tmp_path / "A.mp4"; video.write_bytes(b"x" * 100)
    out_base = tmp_path / "out"
    name = "A"
    d = out_base / name
    (d / "gvhmr" / name).mkdir(parents=True)
    (d / "hamer" / name / "rendered").mkdir(parents=True)
    (d / "emica" / name).mkdir(parents=True)
    (d / "gaze_blink" / name).mkdir(parents=True)

    vid_id = cli.video_identity(video)
    gvhmr_result = d / "gvhmr" / name / "hmr4d_results.pt"
    _write_pt(gvhmr_result)
    cli.stamp_write(gvhmr_result, {"video": vid_id, "static_cam": True, "person": 0})

    hamer_pt = d / "hamer" / name / "rendered" / "hamer_hands.pt"
    if with_hands:
        _write_pt(hamer_pt)
        cli.stamp_write(hamer_pt, {"video": vid_id, "downsample": 1, "batch_size": 48,
                                   "focal": "1000.0"})

    flame = d / "emica" / name / "flame_params.npz"
    np.savez(flame, x=np.zeros(3)); cli.stamp_write(flame, {"video": vid_id})
    gaze = d / "gaze_blink" / name / "gaze_blink.npz"
    np.savez(gaze, x=np.zeros(3)); cli.stamp_write(gaze, {"video": vid_id})

    L = 10

    def _params(path=d / "smplx_params.npz"):
        np.savez(path, num_frames=L,
                 left_hand_valid=np.ones(L, bool), right_hand_valid=np.ones(L, bool),
                 face_valid=np.ones(L, bool), gaze_valid=np.ones(L, bool),
                 ik_coverage=np.float32(0.99))

    calls = []

    def _fake_conda_run(cmd, **kw):
        calls.append(list(cmd))
        if any("merge_body_hands.py" in c for c in cmd):
            _params()                      # merge is the stage that writes the npz
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(cli, "conda_run", _fake_conda_run)
    _params()                              # the previous run's output
    monkeypatch.setattr(cli, "ffprobe_field",
                        lambda v, field, **k: {"r_frame_rate": "25/1", "width": "640",
                                              "height": "480"}.get(field, "10"))
    monkeypatch.setattr(cli, "frame_count", lambda v: L)
    monkeypatch.setattr(cli, "gpu_total_gb", lambda: 80.0)
    monkeypatch.setattr(cli, "track_summary", lambda p: {"n_tracks": 1, "tracks": []})
    monkeypatch.setattr(cli, "multi_person_failure", lambda *a: "")
    monkeypatch.setattr(cli, "spurious_track_warning", lambda *a: "")
    monkeypatch.setattr(cli, "_extract_focal", lambda r: "1000.0")
    monkeypatch.setattr(cli, "hamer_detection_count", lambda p: (5, ""))
    return video, out_base, calls


def _write_pt(path):
    """A real (tiny) torch archive, so integrity checks see whole bytes."""
    import torch
    torch.save({"ok": 1}, path)


def test_no_hands_does_not_reuse_a_previous_runs_hands(tmp_path, monkeypatch, capsys):
    """The summary said 'hands: SKIPPED' while merge and IK ate the last run's hamer_hands.pt.

    cli.py gated merge and IK on the FILESYSTEM, so re-running into the same --output-dir
    with --no-hands silently rebuilt the body from stale hands -- which is exactly what
    the IK_COVERAGE_LOW message tells the user to do after IK has just failed on them.
    """
    from vid2smplx import cli
    video, out_base, calls = _seed_finished_run(tmp_path, monkeypatch)

    args = build_parser().parse_args(["run", str(video), "--output-dir", str(out_base),
                                      "--no-hands", "--skip-doctor"])
    cli.cmd_run(args)

    flat = [" ".join(c) for c in calls]
    assert not any("--hamer_result" in c for c in flat), \
        f"--no-hands still fed hands to merge: {flat}"
    assert not any("ik_hands.py" in c for c in flat), \
        f"--no-hands still ran hand IK: {flat}"
    assert "Step 2/5: Hand estimation (SKIPPED)" in capsys.readouterr().out


def test_hands_are_still_used_when_they_are_wanted(tmp_path, monkeypatch):
    """The negative test above must not be satisfiable by never passing hands at all."""
    from vid2smplx import cli
    video, out_base, calls = _seed_finished_run(tmp_path, monkeypatch)

    args = build_parser().parse_args(["run", str(video), "--output-dir", str(out_base),
                                      "--skip-doctor"])
    cli.cmd_run(args)

    flat = [" ".join(c) for c in calls]
    assert any("--hamer_result" in c for c in flat), flat


# ---- integrity checking must cover .pt, and "cannot verify" is not "intact" ----

def test_zero_byte_pt_with_a_good_stamp_is_not_reused(tmp_path):
    """The two largest artifacts are .pt and used to be exempt from every byte check.

    Measured before the fix: a 0-byte hamer_hands.pt beside a valid stamp -> skip? True.
    """
    from vid2smplx.cli import reuse_or_clear, stamp_path, stamp_write
    for name in ("hamer_hands.pt", "hmr4d_results.pt"):
        art = tmp_path / name
        art.write_bytes(b"")
        stamp_write(art, {"v": 1})
        assert reuse_or_clear(art, {"v": 1}, "HaMeR") is False
        assert not art.exists() and not stamp_path(art).exists()


def test_truncated_pt_is_not_reused(tmp_path):
    import torch
    from vid2smplx.cli import artifact_intact, reuse_or_clear, stamp_write
    art = tmp_path / "hmr4d_results.pt"
    torch.save({"a": torch.zeros(5000)}, art)
    assert artifact_intact(art)[0] == "intact"
    stamp_write(art, {"v": 1})
    assert reuse_or_clear(art, {"v": 1}, "GVHMR") is True          # whole -> reuse

    raw = art.read_bytes(); art.write_bytes(raw[: len(raw) // 2])  # killed mid-write
    state, why = artifact_intact(art)
    assert state == "broken" and "partial" in why
    assert reuse_or_clear(art, {"v": 1}, "GVHMR") is False


def test_cannot_verify_is_a_distinct_signal_from_intact(tmp_path, capsys):
    """The ImportError path used to return the SUCCESS value — silence read as 'verified'."""
    from vid2smplx.cli import artifact_intact, reuse_or_clear, stamp_write
    art = tmp_path / "thing.bin"                     # no integrity check exists for this
    art.write_bytes(b"data")
    state, why = artifact_intact(art)
    assert state == "unknown" and why
    stamp_write(art, {"v": 1})
    assert reuse_or_clear(art, {"v": 1}, "stage") is True          # still reused ...
    assert "integrity NOT checked" in capsys.readouterr().out      # ... but never silently


def test_regenerated_gaze_forces_a_re_merge(tmp_path, monkeypatch, capsys):
    """gaze was fed to merge but left out of merge_key, so a new gaze changed nothing.

    Before: the second run printed `[SKIP]` for merge, so smplx_params.npz kept the OLD
    gaze/blink while summary.json reported coverage read back from the same stale file.
    """
    import numpy as np
    from vid2smplx import cli
    video, out_base, calls = _seed_finished_run(tmp_path, monkeypatch)
    argv = ["run", str(video), "--output-dir", str(out_base), "--skip-doctor"]

    cli.cmd_run(build_parser().parse_args(argv))
    assert any("merge_body_hands.py" in " ".join(c) for c in calls)

    calls.clear()
    cli.cmd_run(build_parser().parse_args(argv))                 # nothing changed -> skip
    assert not any("merge_body_hands.py" in " ".join(c) for c in calls), calls

    gaze = out_base / "A" / "gaze_blink" / "A" / "gaze_blink.npz"
    np.savez(gaze, x=np.ones(3))                                 # gaze regenerated
    calls.clear()
    cli.cmd_run(build_parser().parse_args(argv))
    assert any("merge_body_hands.py" in " ".join(c) for c in calls), \
        "new gaze left merge_key identical — smplx_params.npz would keep the old gaze"


@pytest.mark.parametrize("body", ["[1, 2]", "null", "42", '"hello"'])
def test_a_stamp_that_is_not_an_object_rebuilds_instead_of_crashing(tmp_path, body):
    """Valid JSON, wrong shape: .get() raised AttributeError, which main() does not catch."""
    from vid2smplx.cli import stamp_ok, stamp_path
    art = tmp_path / "x.npz"; art.write_text("q")
    stamp_path(art).write_text(body)
    ok, why = stamp_ok(art, {"v": 1})
    assert ok is False
    assert "not an object" in why and "delete it and re-run" in why


def test_render_after_cleanup_says_how_to_recover(tmp_path):
    """`run --cleanup` then `render` used to die inside render.py with 'Video not found'."""
    import numpy as np
    from vid2smplx.cli import validate_render_args
    d = tmp_path / "A"; d.mkdir()
    np.savez(d / "smplx_params.npz", num_frames=1)            # what --cleanup leaves behind

    a = build_parser().parse_args(["render", str(d), "--layers", "final"])
    errs = []; validate_render_args(a, errs.append)
    assert errs and "--video" in errs[0] and "--cleanup" in errs[0]
    assert str(d) in errs[0]                                   # a command they can paste

    # with --video the source is solved, but 'final' still needs GVHMR's own result
    vid = tmp_path / "v.mp4"; vid.write_bytes(b"x")
    a = build_parser().parse_args(["render", str(d), "--layers", "final", "--video", str(vid)])
    errs = []; validate_render_args(a, errs.append)
    assert errs and "hmr4d_results.pt" in errs[0] and "--layers face,hands" in errs[0]

    # a layer that needs neither is allowed through
    a = build_parser().parse_args(["render", str(d), "--layers", "hands", "--video", str(vid)])
    errs = []; validate_render_args(a, errs.append)
    assert errs == []


def test_render_on_an_untouched_run_dir_still_validates(tmp_path):
    """The new guard must not reject a normal, un-cleaned output dir."""
    import numpy as np
    from vid2smplx.cli import validate_render_args
    d = tmp_path / "A"; (d / "gvhmr" / "A").mkdir(parents=True)
    np.savez(d / "smplx_params.npz", num_frames=1)
    (d / "gvhmr" / "A" / "0_input_video.mp4").write_bytes(b"x")
    (d / "gvhmr" / "A" / "hmr4d_results.pt").write_bytes(b"x")
    a = build_parser().parse_args(["render", str(d), "--layers", "final,global"])
    errs = []; validate_render_args(a, errs.append)
    assert errs == []
