"""Hand loading, filtering, placement, and depth correction."""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from utils import (
    HAMER_FOCAL_LENGTH, HAMER_IMAGE_SIZE,
    MIN_KP_COUNT, MIN_KP_CONF,
)


def load_hand_meshes(params_path: Path | str, num_frames: int) -> dict[int, list[dict]]:
    """Load per-frame MANO hand vertices + cam_t_full.

    Args:
        params_path: Path to hamer_hands.pt (single file) or mano_params/ directory (legacy)
        num_frames: Maximum number of frames

    Returns:
        dict[frame_idx] -> list of {vertices, cam_t_full, is_right, scaled_focal_length}
    """
    params_path = Path(params_path)

    if params_path.suffix == ".pt" and params_path.is_file():
        return _load_hand_meshes_pt(params_path, num_frames)
    elif params_path.is_dir():
        return _load_hand_meshes_npz(params_path, num_frames)
    else:
        print(f"  [WARNING] Hand params not found: {params_path}")
        return defaultdict(list)


def _load_hand_meshes_pt(pt_path, num_frames):
    """Load from consolidated hamer_hands.pt."""
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    by_frame = defaultdict(list)
    n_rejected = 0

    N = len(data["frame_idx"])
    for i in range(N):
        frame_idx = int(data["frame_idx"][i])
        if frame_idx < 0 or frame_idx >= num_frames:
            continue

        kp_count = int(data["kp_count"][i]) if "kp_count" in data else 21
        kp_conf = float(data["kp_mean_conf"][i]) if "kp_mean_conf" in data else 1.0
        if kp_count < MIN_KP_COUNT or kp_conf < MIN_KP_CONF:
            n_rejected += 1
            continue

        by_frame[frame_idx].append({
            "vertices": data["vertices"][i].numpy(),
            "cam_t_full": data["cam_t_full"][i].numpy(),
            "is_right": bool(data["is_right"][i]),
            "scaled_focal_length": float(data["scaled_focal_length"][i]) if "scaled_focal_length" in data else None,
        })

    if n_rejected > 0:
        print(f"  [Confidence] Rejected {n_rejected} low-confidence hand detections")

    return by_frame


def _load_hand_meshes_npz(params_dir, num_frames):
    """Load per-frame MANO hand vertices from NPZ files (legacy format)."""
    params_dir = Path(params_dir)
    by_frame = defaultdict(list)
    n_rejected = 0

    for npz_file in sorted(params_dir.glob("*.npz")):
        match = re.match(r"^(\d+)_(\d+)$", npz_file.stem)
        if not match:
            continue

        frame_idx = int(match.group(1)) - 1
        if frame_idx < 0 or frame_idx >= num_frames:
            continue

        data = dict(np.load(npz_file, allow_pickle=True))
        if "vertices" not in data or "cam_t_full" not in data:
            continue

        kp_count = int(data["kp_count"]) if "kp_count" in data else 21
        kp_conf = float(data["kp_mean_conf"]) if "kp_mean_conf" in data else 1.0
        if kp_count < MIN_KP_COUNT or kp_conf < MIN_KP_CONF:
            n_rejected += 1
            continue

        by_frame[frame_idx].append({
            "vertices": data["vertices"].astype(np.float32),
            "cam_t_full": data["cam_t_full"].astype(np.float32),
            "is_right": bool(data["is_right"]),
            "scaled_focal_length": float(data["scaled_focal_length"]) if "scaled_focal_length" in data else None,
        })

    if n_rejected > 0:
        print(f"  [Confidence] Rejected {n_rejected} low-confidence hand detections")

    return by_frame


def compute_incam_to_global_transform(verts_incam: torch.Tensor, verts_global: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute rigid transform from incam to global via Procrustes on frame 0."""
    vi = verts_incam[0, ::10].float()
    vg = verts_global[0, ::10].float()
    ci = vi.mean(0)
    cg = vg.mean(0)
    H = (vi - ci).T @ (vg - cg)
    U, S, Vt = torch.linalg.svd(H)
    R = Vt.T @ U.T
    if torch.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = cg - R @ ci
    return R, t


def place_hands_dc(hand_meshes_by_frame: dict[int, list[dict]], gvhmr_K: torch.Tensor, smplx_joints_incam: torch.Tensor,
                   smplx_verts_incam: torch.Tensor, smplx_verts_global: torch.Tensor, width: int, height: int, num_frames: int,
                   mano_J: torch.Tensor | None = None, verbose: bool = True) -> tuple[dict[int, dict[str, torch.Tensor]], dict[int, list[dict]]]:
    """Compute depth-corrected wrist targets and hand vertices in incam space.

    Returns:
        dc_wrist_targets: dict[frame_idx] -> {"left": (3,), "right": (3,)}
        hands_incam_dc: dict[frame_idx] -> list of {verts: (778,3), is_right: bool}
    """
    default_hamer_focal = HAMER_FOCAL_LENGTH / HAMER_IMAGE_SIZE * max(width, height)
    gvhmr_fx = gvhmr_K[0, 0].item()

    first_det = next((dets[0] for dets in hand_meshes_by_frame.values() if dets), None)
    hamer_focal_sample = first_det["scaled_focal_length"] if (first_det and first_det["scaled_focal_length"] is not None) else default_hamer_focal
    intrinsics_match = abs(hamer_focal_sample - gvhmr_fx) < 1.0

    WRIST_LEFT = 20
    WRIST_RIGHT = 21

    hands_incam_dc = defaultdict(list)
    dc_wrist_targets = {}

    n_total_dets = 0
    n_wrist_z_negative = 0
    n_cam_t_zero = 0
    n_frame_oob = 0
    n_ok = 0

    for frame_idx, detections in hand_meshes_by_frame.items():
        if frame_idx >= num_frames:
            n_frame_oob += len(detections)
            continue
        for det in detections:
            n_total_dets += 1
            verts = det["vertices"].copy()
            cam_t = det["cam_t_full"]
            is_right = det["is_right"]

            verts[:, 0] = (2 * int(is_right) - 1) * verts[:, 0]

            wrist_idx = WRIST_RIGHT if is_right else WRIST_LEFT
            wrist_z = smplx_joints_incam[frame_idx, wrist_idx, 2].item()

            if wrist_z <= 0:
                n_wrist_z_negative += 1
                continue
            if abs(cam_t[2]) <= 1e-6:
                n_cam_t_zero += 1
                continue

            n_ok += 1

            if not intrinsics_match and det.get("scaled_focal_length") is not None:
                focal_ratio = gvhmr_fx / det["scaled_focal_length"]
                cam_t = cam_t.copy()  # don't modify original
                cam_t[0] *= focal_ratio
                cam_t[1] *= focal_ratio

            scale = wrist_z / cam_t[2]
            verts_scaled = verts * scale
            cam_t_dc = np.array([cam_t[0] * scale, cam_t[1] * scale, wrist_z], dtype=np.float32)

            verts_incam_dc_t = torch.from_numpy(verts_scaled).float() + torch.from_numpy(cam_t_dc).float()

            hands_incam_dc[frame_idx].append({
                "verts": verts_incam_dc_t,
                "is_right": is_right,
            })

            if frame_idx not in dc_wrist_targets:
                dc_wrist_targets[frame_idx] = {}
            key = "right" if is_right else "left"
            # Use actual MANO wrist joint position, not mesh center (cam_t_dc)
            if mano_J is not None:
                dc_wrist_targets[frame_idx][key] = (mano_J[0] @ verts_incam_dc_t).detach()
            else:
                dc_wrist_targets[frame_idx][key] = torch.from_numpy(cam_t_dc).float()

    if verbose:
        n_unique_frames = len(dc_wrist_targets)
        print(f"  [DC Stats] {n_ok}/{n_total_dets} detections passed -> {n_unique_frames} frames with targets")
        if n_wrist_z_negative > 0:
            print(f"  [DC Stats] Rejected: wrist_z<=0: {n_wrist_z_negative}")
        if n_cam_t_zero > 0:
            print(f"  [DC Stats] Rejected: cam_t_z~0: {n_cam_t_zero}")
        if n_frame_oob > 0:
            print(f"  [DC Stats] Rejected: frame out of bounds: {n_frame_oob}")
        print(f"  [DC Stats] intrinsics_match={intrinsics_match} (hamer={hamer_focal_sample:.1f}, gvhmr={gvhmr_fx:.1f})")

    return dc_wrist_targets, hands_incam_dc
