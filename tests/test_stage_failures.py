"""A stage that crashed must not look like a stage that found nothing.

Both confusions shipped: `run_hamer_video.py` printed "HaMeR inference failed!" and returned
(exit 0), so the CLI reported "[OK] Hand params: <path that does not exist>"; and EMICA reused
the last detected bbox forever, so one detection at frame 0 marked a whole video face_valid.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"


def _load(name: str):
    """Import a scripts/*.py module (they do `from utils import ...`, so scripts/ must be on the path)."""
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(f"_v2s_{name}", SCRIPTS / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(str(SCRIPTS))


# ---- a crashed stage exits non-zero ----

def test_hamer_failure_exits_non_zero(tmp_path, monkeypatch):
    mod = _load("run_hamer_video")
    monkeypatch.setattr(mod, "run_hamer_inference", lambda *a, **k: False)
    monkeypatch.setattr(sys, "argv", ["run_hamer_video.py", "--video", str(tmp_path / "v.mp4"),
                                      "--out_folder", str(tmp_path / "out")])
    with pytest.raises(SystemExit) as e:
        mod.main()
    assert e.value.code not in (0, None), "a HaMeR crash exited 0; the caller then prints [OK]"


def test_gaze_blink_unreadable_video_exits_non_zero(tmp_path):
    """End to end through the real interpreter: rc, not just a SystemExit object."""
    bogus = tmp_path / "not_a_video.mp4"
    bogus.write_bytes(b"this is not a video")
    r = subprocess.run([sys.executable, str(SCRIPTS / "run_gaze_blink.py"),
                        "--video", str(bogus), "--out_folder", str(tmp_path / "out")],
                       capture_output=True, text=True, cwd=str(SCRIPTS))
    assert r.returncode != 0, f"unreadable video exited 0:\n{r.stdout}\n{r.stderr}"


def test_cli_never_prints_ok_for_a_missing_path():
    """The [OK] lines for hands and gaze must be guarded by the artifact actually existing."""
    src = (REPO / "vid2smplx" / "cli.py").read_text()
    assert 'if hands_present():\n            print(f"  [OK] Hand params: {hamer_params}")' in src
    assert 'if gaze_blink_result.exists():\n            print(f"  [OK] Gaze+Blink:' in src


# ---- EMICA must not fabricate faces ----

class _FakeCap:
    """cv2.VideoCapture over a list of (detected?) frames."""

    def __init__(self, detections):
        self.detections, self.i = detections, 0

    def isOpened(self):
        return True

    def get(self, _prop):
        return float(len(self.detections))

    def read(self):
        if self.i >= len(self.detections):
            return False, None
        self.i += 1
        return True, np.zeros((64, 64, 3), np.uint8)

    def release(self):
        pass


def _fake_mediapipe(detections):
    """A mediapipe stand-in whose detector fires only on the frames in `detections`."""
    import types

    class _BB:
        xmin = ymin = 0.25
        width = height = 0.5

    class _Det:
        location_data = types.SimpleNamespace(relative_bounding_box=_BB())

    class _FD:
        def __init__(self, **_kw):
            self.i = -1

        def process(self, _img):
            self.i += 1
            return types.SimpleNamespace(detections=[_Det()] if detections[self.i] else None)

        def close(self):
            pass

    return types.SimpleNamespace(
        solutions=types.SimpleNamespace(face_detection=types.SimpleNamespace(FaceDetection=_FD)))


def _run_detect(monkeypatch, detections, tmp_path):
    mod = _load("run_emica")
    monkeypatch.setitem(sys.modules, "mediapipe", _fake_mediapipe(detections))
    monkeypatch.setattr(mod.cv2, "VideoCapture", lambda _p: _FakeCap(detections))
    monkeypatch.chdir(tmp_path)
    return mod, mod.detect_and_crop_streaming("fake.mp4")


def test_carried_forward_bbox_is_capped_and_marked_stale(monkeypatch, tmp_path):
    """One detection at frame 0 used to mark all 30 frames as faces."""
    mod, (_chunks, _n, _bboxes, valid, stale) = _run_detect(
        monkeypatch, [True] + [False] * 29, tmp_path)
    assert mod.MAX_BBOX_CARRY_FORWARD == 5
    assert valid == list(range(1 + mod.MAX_BBOX_CARRY_FORWARD)), \
        "the stale bbox was carried past its cap"
    assert stale == [False] + [True] * mod.MAX_BBOX_CARRY_FORWARD


def test_a_fresh_detection_resets_the_carry_forward(monkeypatch, tmp_path):
    dets = [True, False, False, True, False]
    _mod, (_c, _n, _b, valid, stale) = _run_detect(monkeypatch, dets, tmp_path)
    assert valid == [0, 1, 2, 3, 4]
    assert stale == [False, True, True, False, True]


def _fake_gvhmr(tmp_path, n=10):
    import torch
    p = tmp_path / "hmr4d_results.pt"
    torch.save({"smpl_params_global": {"body_pose": torch.zeros(n, 63), "global_orient": torch.zeros(n, 3),
                                       "betas": torch.zeros(n, 10), "transl": torch.zeros(n, 3)},
                "K_fullimg": torch.eye(3)}, p)
    return p


def _flame_npz(tmp_path, timesteps, stale=None):
    n = len(timesteps)
    d = {"expr": np.zeros((n, 100), np.float32), "jaw_pose": np.zeros((n, 3), np.float32),
         "eyes_pose": np.zeros((n, 6), np.float32),
         "timestep_id": np.array(timesteps, np.int64)}
    if stale is not None:
        d["bbox_stale"] = np.array(stale, bool)
    p = tmp_path / "flame_params.npz"
    np.savez(p, **d)
    return p


def test_stale_frames_are_not_marked_face_valid(tmp_path):
    """merge turns timestep_id into face_valid; a carried-forward crop must not count."""
    mod = _load("merge_body_hands")
    out = mod.merge(_fake_gvhmr(tmp_path), None, tmp_path / "smplx.npz",
                    flame_result=_flame_npz(tmp_path, [0, 1, 2], [False, True, True]))
    assert list(out["face_valid"][:3]) == [True, False, False]
    assert out["face_valid"].sum() == 1


def test_a_golden_without_bbox_stale_still_merges(tmp_path):
    """Outputs made before the key existed must keep loading, all frames valid as before."""
    mod = _load("merge_body_hands")
    out = mod.merge(_fake_gvhmr(tmp_path), None, tmp_path / "smplx.npz",
                    flame_result=_flame_npz(tmp_path, [0, 1, 2]))
    assert list(out["face_valid"][:3]) == [True, True, True]
