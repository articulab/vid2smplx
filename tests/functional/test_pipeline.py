"""Functional tests: run the real models on a short seeded clip and check every stage.

    pytest -m functional tests/functional -x -s          # from inside the env, with weights present
    pytest -m functional tests/functional --update-golden   # accept the current output as reference

Outputs (renders, contact sheet, curves) land in tests/functional/out/ for eyeballing.
Seeding makes the pipeline deterministic on one machine (~1e-7 run to run) but NOT across machines
(~0.027 rad / ~0.088 betas measured), so the golden carries its provenance and test_golden branches on it.
See vid2smplx/provenance.py and docs/install.md.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from vid2smplx.provenance import (PROVENANCE_KEY, capture_environment, code_differences,
                                  compare_environment, describe, golden_unusable, mismatch_message,
                                  read_provenance, verdict, write_golden)

pytestmark = pytest.mark.functional

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "out"
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
# One golden per (clip, length): the fast default (clip_talking, 1.5 s, 38 frames) is the
# everyday regression check; clip_dancing at 20 s exercises the arms and the IK path, which a
# talking head never does. Derived from the clip stem so any --clip/--seconds pair gets its own.
def golden_for(clip: Path) -> Path:
    return GOLDEN_DIR / f"{clip.stem}.npz"
SEED = 0
# (key, trailing shape) — T is the frame count
NPZ_SPEC = {
    "body_pose": (63,), "global_orient": (3,), "transl": (3,), "betas": (10,),
    "left_hand_pose": (45,), "right_hand_pose": (45,),
    "jaw_pose": (3,), "expression": (100,), "leye_pose": (3,), "reye_pose": (3,),
    "gaze_pitch": (), "gaze_yaw": (), "blink_left": (), "blink_right": (),
    "left_hand_valid": (), "right_hand_valid": (), "face_valid": (), "gaze_valid": (),
    "K_fullimg": (3, 3),
}
# tolerance for golden comparison, per key family (radians / metres / unitless).
# Used when the golden was produced in THIS environment; run-to-run noise there is ~1e-7.
TOL = {"transl": 2e-2, "betas": 5e-2, "gaze": 5e-2, "blink": 5e-2, "expression": 0.1, "default": 2e-2}
# Envelope for a golden from a DIFFERENT environment. Measured cross-machine drift with no code
# change: ~0.027 rad on non-arm body_pose, ~0.088 on betas. Values are ~2x that, so anything over
# is too big to blame on hardware; the un-IK'd-arms regression (0.54 rad) is far above either.
TOL_CROSS_ENV = {"transl": 6e-2, "betas": 0.2, "gaze": 0.1, "blink": 0.1, "expression": 0.3, "default": 6e-2}


def _ffprobe_frames(video: Path) -> int:
    out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                                   "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(video)], text=True)
    return int(out.strip().splitlines()[0])


@pytest.fixture(scope="session")
def clip(request) -> Path:
    src = REPO / request.config.getoption("--clip")
    secs = request.config.getoption("--seconds")
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / f"{src.stem}_{secs:g}s.mp4"
    if not dst.exists():
        subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-t", str(secs),
                               "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-an", str(dst)])
    return dst


@pytest.fixture(scope="session")
def run(clip) -> Path:
    """Run the whole pipeline once (full debug renders, fixed seed). Cached: delete tests/functional/out to rerun."""
    out_dir = OUT / "output" / clip.stem
    if not (out_dir / "SUCCESS").exists():
        shutil.rmtree(out_dir, ignore_errors=True)
        subprocess.check_call([sys.executable, "-m", "vid2smplx.cli", "run", str(clip),
                               "--full-debug", "--seed", str(SEED), "--output-dir", str(OUT / "output"),
                               "--skip-doctor"], cwd=REPO)
    assert (out_dir / "SUCCESS").exists()
    return out_dir


@pytest.fixture(scope="session")
def T(clip) -> int:
    return _ffprobe_frames(clip)


@pytest.fixture(scope="session")
def params(run) -> dict:
    return dict(np.load(run / "smplx_params.npz", allow_pickle=True))


def test_gvhmr_body(run, clip, T):
    import torch
    pred = torch.load(run / "gvhmr" / clip.stem / "hmr4d_results.pt", map_location="cpu", weights_only=False)
    body = pred["smpl_params_global"]["body_pose"]
    assert body.shape == (T, 63), body.shape
    assert torch.isfinite(body).all()
    assert body.abs().max() < np.pi + 0.1   # axis-angle sanity


def test_hamer_hands(run, clip):
    import torch
    pt = run / "hamer" / clip.stem / "rendered" / "hamer_hands.pt"
    assert pt.exists(), "HaMeR produced no hands"
    hands = torch.load(pt, map_location="cpu", weights_only=False)
    assert len(hands["frame_idx"]) > 0


def test_emica_face(run, clip, T):
    d = np.load(run / "emica" / clip.stem / "flame_params.npz", allow_pickle=True)
    exp = d["expr"]
    assert exp.shape[0] == T, exp.shape
    assert np.isfinite(exp).all()


def test_gaze_blink(run, clip, T):
    d = np.load(run / "gaze_blink" / clip.stem / "gaze_blink.npz", allow_pickle=True)
    assert d["gaze_pitch"].shape == (T,)
    assert 0.0 < np.nanmedian(d["blink_left"]) < 1.0   # EAR in a sane range


def test_merged_npz_shapes(params, T):
    assert int(params["num_frames"]) == T
    for key, tail in NPZ_SPEC.items():
        assert key in params, f"missing {key}"
        assert params[key].shape == (T, *tail), f"{key}: {params[key].shape} != {(T, *tail)}"
        assert np.isfinite(params[key].astype(float)).all(), f"{key} has NaN/inf"
    # the talking clip has a visible face and hands in every frame
    assert params["face_valid"].mean() > 0.8
    assert (params["left_hand_valid"] | params["right_hand_valid"]).mean() > 0.5


def test_ik_applied(params):
    assert "ik_wrist_loss" in params, "IK step did not run"
    loss = params["ik_wrist_loss"]
    covered = loss[loss > 0]
    assert covered.size / loss.size >= 0.2
    assert np.median(covered) < 0.05, f"median wrist error {np.median(covered):.3f} m"


def test_renders_exist(run):
    for rel in ["render/final_incam.mp4", "render/hands_incam.mp4", "render/face_incam.mp4"]:
        p = run / rel
        assert p.exists() and p.stat().st_size > 10_000, rel


def test_visualize(run, params, T):
    """Contact sheet + parameter curves for eyeballing. Always passes if it can write the files."""
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = np.linspace(0, T - 1, 6).astype(int)
    rows = []
    for rel in ["render/final_incam.mp4", "render/hands_incam.mp4", "render/face_incam.mp4"]:
        cap = cv2.VideoCapture(str(run / rel))
        frames = []
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if ok:
                frames.append(cv2.resize(f, (320, int(320 * f.shape[0] / f.shape[1]))))
        if frames:
            rows.append(np.concatenate(frames, axis=1))
    h = min(r.shape[0] for r in rows)
    sheet = np.concatenate([r[:h] for r in rows], axis=0)
    cv2.imwrite(str(OUT / "contact_sheet.png"), sheet)

    fig, ax = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    ax[0].plot(params["body_pose"][:, :9]); ax[0].set_title("body_pose (first 3 joints)")
    ax[1].plot(params["right_hand_pose"][:, :9]); ax[1].set_title("right_hand_pose (first 3 joints)")
    ax[2].plot(params["jaw_pose"]); ax[2].plot(params["expression"][:, :3], "--"); ax[2].set_title("jaw_pose / expression[:3]")
    ax[3].plot(params["gaze_pitch"], label="pitch"); ax[3].plot(params["gaze_yaw"], label="yaw"); ax[3].legend(); ax[3].set_title("gaze (rad)")
    ax[4].plot(params["blink_left"], label="L"); ax[4].plot(params["blink_right"], label="R"); ax[4].legend(); ax[4].set_title("blink EAR")
    fig.tight_layout(); fig.savefig(OUT / "curves.png", dpi=100)
    print(f"\n  eyeball: {OUT/'contact_sheet.png'}  {OUT/'curves.png'}  {run/'render/final_incam.mp4'}")


def _compare(params, ref, tol_table) -> list:
    bad = []
    for key in NPZ_SPEC:
        tol = next((v for k, v in tol_table.items() if k in key), tol_table["default"])
        a, b = params[key].astype(float), ref[key].astype(float)
        if key.endswith("_valid"):
            agree = (a == b).mean()
            if agree < 0.95:
                bad.append(f"{key}: validity masks agree on {agree:.0%}")
            continue
        mask = None
        if key.startswith("left_hand"):
            mask = ref["left_hand_valid"]
        elif key.startswith("right_hand"):
            mask = ref["right_hand_valid"]
        elif key.startswith("gaze"):
            mask = ref["gaze_valid"]
        elif key.startswith(("jaw", "expression", "flame", "leye", "reye")):
            mask = ref["face_valid"]
        if mask is not None:
            a, b = a[mask.astype(bool)], b[mask.astype(bool)]
        err = np.abs(a - b).max() if a.size else 0.0
        if err > tol:
            bad.append(f"{key}: max abs diff {err:.4f} > {tol}")
    return bad


def test_golden(request, params, run, clip):
    """Compare against the accepted reference, at a tolerance that depends on the golden's provenance."""
    GOLDEN = golden_for(clip)
    env = capture_environment(REPO)
    if request.config.getoption("--update-golden"):
        write_golden(run / "smplx_params.npz", GOLDEN, env)
        pytest.skip(f"golden updated: {GOLDEN}\n  provenance: {describe(env)}")
    if not GOLDEN.exists():
        pytest.skip(f"no golden yet — eyeball {OUT} then rerun with --update-golden")
    ref = dict(np.load(GOLDEN, allow_pickle=True))
    assert int(ref["num_frames"]) == int(params["num_frames"])
    golden_env = read_provenance(ref)
    # Before comparing numbers: a golden of unknown origin makes every verdict below
    # unreachable except 'skip'. Say so and fail, rather than pass silently.
    unusable = golden_unusable(golden_env)
    if unusable:
        pytest.fail(unusable)
    env_diffs = compare_environment(golden_env, env)

    bad_tight = _compare(params, ref, TOL)
    bad_wide = _compare(params, ref, TOL_CROSS_ENV) if env_diffs else []
    call = verdict(env_diffs, bad_tight, bad_wide)

    if not env_diffs:
        code = code_differences(golden_env, env) or ["none detected"]
        note = (f"\n(same environment: {describe(env)}; run-to-run noise here is ~1e-7, so this is a\n"
                "code change, not hardware. Code delta: " + "; ".join(code) + ")")
        assert call == "pass", "\n".join(bad_tight) + note
        return
    msg = mismatch_message(golden_env, env, env_diffs,
                           (bad_wide if call == "fail" else bad_tight) or ["(none — inside the tight tolerance anyway)"],
                           over_envelope=call == "fail")
    if call == "fail":
        pytest.fail(msg)
    pytest.skip(msg)
