"""Two non-finite paths that used to be silent.

1. `rotmat_to_axis_angle` is Rodrigues' formula, whose axis term is 0/0 at a half turn:
   at exactly 180 degrees it returned the ZERO vector — the identity — for a rotation that
   is as far from the identity as a rotation gets, and just short of 180 degrees it
   returned noise. HaMeR hand rotation matrices take this path
   (scripts/merge_body_hands.py: rotmat_batch_to_axis_angle).
2. A non-finite IK target poisons Adam's moment buffers, and the temporal smoothness term
   spreads it one frame per step until the whole sequence is nan — and it was written out
   with `ik_wrist_loss`, which is exactly what scripts/render.py treats as "this file was
   IK'd".
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"


def _load(name: str):
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(f"_v2s_{name}", SCRIPTS / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(str(SCRIPTS))


utils = _load("utils")
ik_hands = _load("ik_hands")


def rotmat_of(axis_angle: np.ndarray) -> np.ndarray:
    """Independent inverse (Rodrigues' forward direction has no singularity)."""
    aa = np.asarray(axis_angle, dtype=np.float64)
    theta = np.linalg.norm(aa)
    if theta < 1e-12:
        return np.eye(3)
    k = aa / theta
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * kx + (1 - np.cos(theta)) * (kx @ kx)


def roundtrip_err(axis_angle: np.ndarray) -> float:
    r = rotmat_of(axis_angle)
    out = utils.rotmat_to_axis_angle(r)
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out)), f"non-finite axis-angle {out} for {axis_angle}"
    return float(np.abs(rotmat_of(out.astype(np.float64)) - r).max())


# ---- the half turn ----

@pytest.mark.parametrize("axis", [(1, 0, 0), (0, 1, 0), (0, 0, 1),
                                  (1, 2, -3), (-0.3, 0.7, 0.1)])
def test_half_turn_roundtrips(axis):
    u = np.asarray(axis, dtype=np.float64)
    u /= np.linalg.norm(u)
    assert roundtrip_err(np.pi * u) < 1e-5


@pytest.mark.parametrize("gap", [1e-2, 1e-3, 1e-4, 1e-5, 1e-7, 1e-9, 0.0])
def test_near_half_turn_neighbourhood(gap):
    """Both sides of the branch boundary, including where Rodrigues used to be noise."""
    u = np.array([0.3, -0.5, 0.81])
    u /= np.linalg.norm(u)
    assert roundtrip_err((np.pi - gap) * u) < 1e-5


@pytest.mark.parametrize("theta", [0.0, 1e-12, 1e-7, 1e-3, 0.5, 2.0, 3.0])
def test_small_and_ordinary_angles(theta):
    u = np.array([0.1, 0.9, -0.42])
    u /= np.linalg.norm(u)
    # theta < 1e-6 is the pre-existing identity guard: it returns zeros, so the recovered
    # rotation differs from the input by at most theta itself.
    assert roundtrip_err(theta * u) < max(1e-5, 2 * theta)


def test_identity_is_zero():
    assert np.array_equal(utils.rotmat_to_axis_angle(np.eye(3)), np.zeros(3, np.float32))


def test_fuzz_random_rotations_roundtrip():
    rng = np.random.default_rng(20260913)
    worst = 0.0
    for _ in range(2000):
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        # Sample the whole range including the pi end, where the old code failed.
        theta = rng.choice([rng.uniform(0, np.pi), np.pi - abs(rng.normal(0, 1e-4))])
        worst = max(worst, roundtrip_err(theta * u))
    assert worst < 1e-5, worst


def test_batch_wrapper_is_finite_on_half_turns():
    mats = np.stack([rotmat_of(np.pi * np.eye(3)[i]) for i in range(3)])
    out = utils.rotmat_batch_to_axis_angle(mats)
    assert out.shape == (3, 3)
    assert np.all(np.isfinite(out))
    for i in range(3):
        assert np.abs(rotmat_of(out[i].astype(np.float64)) - mats[i]).max() < 1e-5


# ---- IK target guard ----

def _targets(num_frames=4, n_ring=2):
    rt_l = torch.zeros(num_frames, n_ring, 3)
    rt_r = torch.zeros(num_frames, n_ring, 3)
    rw_l = torch.ones(num_frames, 1)
    rw_r = torch.ones(num_frames, 1)
    body = torch.zeros(num_frames, 21, 3)
    frames = torch.tensor([10, 11, 12, 13])
    return rt_l, rt_r, rw_l, rw_r, body, frames


def test_finite_targets_pass():
    ik_hands.assert_targets_finite(*_targets())


def test_nan_target_names_the_frame():
    rt_l, rt_r, rw_l, rw_r, body, frames = _targets()
    rt_r[2, 1, 0] = float("nan")
    with pytest.raises(RuntimeError) as e:
        ik_hands.assert_targets_finite(rt_l, rt_r, rw_l, rw_r, body, frames)
    msg = str(e.value)
    assert "12" in msg                       # the original frame index, not the IK index
    assert "HaMeR" in msg and "--no-hands" in msg


def test_inf_in_body_pose_is_caught():
    rt_l, rt_r, rw_l, rw_r, body, frames = _targets()
    body[0, 17, 2] = float("inf")
    with pytest.raises(RuntimeError, match="10"):
        ik_hands.assert_targets_finite(rt_l, rt_r, rw_l, rw_r, body, frames)


def test_unweighted_frame_is_not_a_target():
    """A frame with weight 0 contributes nothing to the loss, so its slot is not a target."""
    rt_l, rt_r, rw_l, rw_r, body, frames = _targets()
    rw_l[1] = 0.0
    rw_r[1] = 0.0
    rt_l[1] = float("nan")
    rt_r[1] = float("nan")
    ik_hands.assert_targets_finite(rt_l, rt_r, rw_l, rw_r, body, frames)


# ---- save guard ----

def _save_data(num_frames=3):
    return {"body_pose": np.zeros((num_frames, 63), np.float32),
            "ik_wrist_loss": np.zeros(num_frames, np.float32),
            "betas": np.zeros(10, np.float32)}


def test_finite_solution_is_written(tmp_path):
    path = tmp_path / "smplx_params.npz"
    ik_hands.save_ik_npz(path, _save_data())
    assert path.exists()
    assert "ik_wrist_loss" in np.load(path).files


def test_nan_solution_is_not_written(tmp_path):
    path = tmp_path / "smplx_params.npz"
    data = _save_data()
    data["body_pose"][1, 5] = float("nan")
    with pytest.raises(RuntimeError, match="refusing to save"):
        ik_hands.save_ik_npz(path, data)
    assert not path.exists()


def test_nan_loss_does_not_overwrite_an_existing_file(tmp_path):
    path = tmp_path / "smplx_params.npz"
    ik_hands.save_ik_npz(path, _save_data())
    before = path.read_bytes()

    data = _save_data()
    data["body_pose"][:] = 1.0
    data["ik_wrist_loss"][2] = float("nan")
    with pytest.raises(RuntimeError):
        ik_hands.save_ik_npz(path, data)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp*"))


# --- the torch twin of the numpy near-pi bug -------------------------------------------------
# matrix_to_axis_angle canonicalised with torch.sign(w). sign(0) == 0, so at theta = pi
# (w = 0) the quaternion was zeroed and the function returned the identity: a 180-degree
# rotation silently became no rotation. Reached on the default path from run_emica.py:201
# via rot6d_to_axis_angle.
@pytest.mark.parametrize("diag, axis", [
    ((1.0, -1.0, -1.0), 0),
    ((-1.0, 1.0, -1.0), 1),
    ((-1.0, -1.0, 1.0), 2),
])
def test_matrix_to_axis_angle_represents_a_half_turn(diag, axis):
    torch = pytest.importorskip("torch")
    from utils import matrix_to_axis_angle
    R = torch.tensor(np.diag(np.array(diag, dtype=np.float64))[None], dtype=torch.float32)
    got = matrix_to_axis_angle(R)[0].numpy()
    assert np.isfinite(got).all()
    # a half turn has magnitude pi about the one axis whose diagonal entry stayed +1
    assert abs(np.linalg.norm(got) - np.pi) < 1e-5, f"half turn collapsed to {got}"
    assert abs(abs(got[axis]) - np.pi) < 1e-5, f"half turn on the wrong axis: {got}"
