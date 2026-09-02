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


def test_missing_ik_fails():
    """ik_coverage absent means the IK stage never ran — arms are raw GVHMR."""
    with tempfile.TemporaryDirectory() as t:
        qc = quality_report(_params(t, hl=0.9, hr=0.9, ik=False), {}, n_hand_det=9999)
        assert any("ik_coverage" in f for f in qc["failures"]), qc
