"""The gate must tell 'HaMeR crashed' apart from 'hands were not visible'.

Coverage alone cannot: both look like a low percentage. Only the first is a bug,
and conflating them once shipped 59 clips with no hands at all.
"""
import tempfile
from pathlib import Path

import numpy as np

from vid2smplx.cli import quality_report


def _params(tmp, L=100, hl=0.0, hr=0.0, ik=True):
    p = Path(tmp) / "p.npz"
    d = dict(num_frames=L,
             left_hand_valid=np.arange(L) < int(L * hl),
             right_hand_valid=np.arange(L) < int(L * hr),
             face_valid=np.ones(L, bool),
             gaze_valid=np.ones(L, bool))
    if ik:
        d["ik_coverage"] = np.float32(0.99)
    np.savez(p, **d)
    return p


def test_hamer_crash_fails():
    """No HaMeR output file at all -> the hand stage failed."""
    with tempfile.TemporaryDirectory() as t:
        qc = quality_report(_params(t), {}, n_hand_det=None)
        assert qc["failures"]


def test_occluded_hands_are_kept():
    """Hands simply not visible: fallback pose is fine, warn but do not fail."""
    with tempfile.TemporaryDirectory() as t:
        qc = quality_report(_params(t, hl=0.04, hr=0.33), {}, n_hand_det=1234)
        assert not qc["failures"], qc["failures"]
        assert qc["warnings"]


def test_healthy_clip_is_clean():
    with tempfile.TemporaryDirectory() as t:
        qc = quality_report(_params(t, hl=0.99, hr=0.99), {}, n_hand_det=9999)
        assert not qc["failures"] and not qc["warnings"], qc


def test_disabled_stages_are_skipped_not_failed():
    """--no-hands/--no-face: the absent stage is the user's intent, not a failure."""
    with tempfile.TemporaryDirectory() as t:
        p = _params(t, ik=False)
        np.savez(p, **{**dict(np.load(p, allow_pickle=True)),
                       "face_valid": np.zeros(100, bool), "gaze_valid": np.zeros(100, bool)})
        qc = quality_report(p, {}, n_hand_det=None, hands=False, face=False)
        assert not qc["failures"], qc["failures"]
        assert not qc["warnings"], qc["warnings"]
        assert qc["stages"] == {"hands": "SKIPPED", "face": "SKIPPED", "gaze": "SKIPPED"}


def test_missing_ik_fails():
    """ik_coverage absent means the IK stage never ran — arms are raw GVHMR."""
    with tempfile.TemporaryDirectory() as t:
        qc = quality_report(_params(t, hl=0.9, hr=0.9, ik=False), {}, n_hand_det=9999)
        assert any("ik_coverage" in f for f in qc["failures"]), qc


def test_skipped_stages_report_no_coverage_numbers():
    """--no-hands printed 'hands_left: 0.0' next to 'hands: SKIPPED' — it reads as a failure."""
    with tempfile.TemporaryDirectory() as t:
        qc = quality_report(_params(t, ik=False), {}, n_hand_det=None, hands=False, face=True,
                            gaze=True)
        assert "hands_left" not in qc and "hands_right" not in qc
        assert qc["face"] == 1.0 and qc["gaze"] == 1.0
        qc = quality_report(_params(t), {}, n_hand_det=5, hands=True, face=False)
        assert "face" not in qc and "gaze" not in qc
        assert "hands_left" in qc


def test_absent_and_unreadable_hamer_output_are_different_faults():
    """'no output file' and 'file exists but will not load' need different messages."""
    from vid2smplx.cli import hamer_detection_count
    with tempfile.TemporaryDirectory() as t:
        missing = Path(t) / "nope.pt"
        n, why = hamer_detection_count(missing)
        assert n is None and "produced no output file" in why

        broken = Path(t) / "hamer_hands.pt"
        broken.write_bytes(b"not a torch file")
        n, why = hamer_detection_count(broken)
        assert n is None and "cannot be read" in why and "Delete it" in why
        assert "produced no output file" not in why

        qc = quality_report(_params(t, hl=0.9, hr=0.9), {}, n_hand_det=n, hand_failure=why)
        assert qc["failures"] == [why]


def test_multi_person_warning_reaches_the_summary():
    with tempfile.TemporaryDirectory() as t:
        qc = quality_report(_params(t, hl=0.9, hr=0.9), {}, n_hand_det=9,
                            multi_person="2 people are in this video")
        assert any("2 people" in w for w in qc["warnings"])
