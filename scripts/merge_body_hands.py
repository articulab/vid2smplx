#!/usr/bin/env python3
"""Merge GVHMR body + HaMeR hands + FLAME face + gaze/blink into SMPL-X params.

Output: smplx_params.npz with unified body, hands, face, gaze, and blink.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch

from utils import (
    rotmat_to_axis_angle, rotmat_batch_to_axis_angle,
    load_mano_mean_hand_pose, MIN_KP_COUNT, MIN_KP_CONF,
)


# ---------------------------------------------------------------------------
# Quaternion / SLERP utilities for hand interpolation
# ---------------------------------------------------------------------------

def _aa_to_quat(aa):
    """Axis-angle (3,) -> quaternion (4,) [w,x,y,z]."""
    angle = np.linalg.norm(aa)
    if angle < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    axis = aa / angle
    ha = angle / 2
    return np.array([np.cos(ha), *(axis * np.sin(ha))], dtype=np.float32)


def _quat_to_aa(q):
    """Quaternion (4,) [w,x,y,z] -> axis-angle (3,)."""
    w = q[0]
    v = q[1:]
    sin_ha = np.linalg.norm(v)
    if sin_ha < 1e-8:
        return np.zeros(3, dtype=np.float32)
    axis = v / sin_ha
    angle = 2.0 * np.arctan2(sin_ha, w)
    return (axis * angle).astype(np.float32)


def _slerp(q1, q2, t):
    """Spherical linear interpolation between two quaternions."""
    dot = np.dot(q1, q2)
    if dot < 0:
        q2 = -q2
        dot = -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:
        return q1 + t * (q2 - q1)
    theta = np.arccos(dot)
    return (np.sin((1 - t) * theta) * q1 + np.sin(t * theta) * q2) / np.sin(theta)


def slerp_hand_pose(pose1, pose2, t):
    """SLERP interpolation between two hand poses (45,) in axis-angle."""
    result = np.zeros(45, dtype=np.float32)
    for j in range(15):
        q1 = _aa_to_quat(pose1[j*3:(j+1)*3])
        q2 = _aa_to_quat(pose2[j*3:(j+1)*3])
        q_interp = _slerp(q1, q2, t)
        result[j*3:(j+1)*3] = _quat_to_aa(q_interp)
    return result


def interpolate_hand_gaps(hand_pose: np.ndarray, hand_valid: np.ndarray, fallback_pose: np.ndarray, max_gap: int = 30) -> np.ndarray:
    """Fill invalid frames with SLERP from nearest valid neighbors."""
    L = len(hand_valid)
    result = hand_pose.copy()

    i = 0
    while i < L:
        if hand_valid[i]:
            i += 1
            continue

        gap_start = i
        while i < L and not hand_valid[i]:
            i += 1
        gap_end = i

        gap_len = gap_end - gap_start
        prev_idx = gap_start - 1 if gap_start > 0 and hand_valid[gap_start - 1] else None
        next_idx = gap_end if gap_end < L and hand_valid[gap_end] else None

        for fi in range(gap_start, gap_end):
            if prev_idx is not None and next_idx is not None and gap_len <= max_gap:
                t = (fi - gap_start + 1) / (gap_len + 1)
                result[fi] = slerp_hand_pose(result[prev_idx], result[next_idx], t)
            elif prev_idx is not None and gap_len <= max_gap:
                t = min(1.0, (fi - gap_start + 1) / (max_gap + 1))
                result[fi] = slerp_hand_pose(result[prev_idx], fallback_pose, t)
            elif next_idx is not None and gap_len <= max_gap:
                t = min(1.0, (gap_end - fi) / (max_gap + 1))
                result[fi] = slerp_hand_pose(fallback_pose, result[next_idx], t)
            else:
                result[fi] = fallback_pose

    return result


# ---------------------------------------------------------------------------
# GVHMR / HaMeR loading
# ---------------------------------------------------------------------------

def load_gvhmr(result_path: Path | str) -> dict[str, dict[str, np.ndarray] | np.ndarray]:
    """Load GVHMR results from .pt file."""
    data = torch.load(result_path, map_location="cpu", weights_only=False)

    params = {}
    for coord_key in ["smpl_params_global", "smpl_params_incam"]:
        if coord_key in data:
            p = data[coord_key]
            params[coord_key] = {
                k: v.numpy() if torch.is_tensor(v) else np.array(v)
                for k, v in p.items()
            }

    if "K_fullimg" in data:
        K = data["K_fullimg"]
        params["K_fullimg"] = K.numpy() if torch.is_tensor(K) else np.array(K)

    return params


def load_hamer_params(params_path: Path | str) -> dict[str, list[dict]]:
    """Load HaMeR hand params from .pt or legacy mano_params/ directory.

    Returns dict[frame_name] -> list of hand detection dicts.
    """
    params_path = Path(params_path)
    if params_path.suffix == ".pt" and params_path.is_file():
        return _load_hamer_params_pt(params_path)
    elif params_path.is_dir():
        return _load_hamer_params_npz(params_path)
    else:
        print(f"  [WARNING] HaMeR params not found: {params_path}")
        return {}


def _load_hamer_params_pt(pt_path):
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    frames = {}
    N = len(data["frame_idx"])
    for i in range(N):
        frame_idx_0 = int(data["frame_idx"][i])
        frame_name = f"{frame_idx_0 + 1:06d}"

        det = {
            "hand_pose": data["hand_pose"][i].numpy(),
            "global_orient": data["global_orient"][i].numpy(),
            "betas": data["betas"][i].numpy(),
            "vertices": data["vertices"][i].numpy(),
            "cam_t_full": data["cam_t_full"][i].numpy(),
            "is_right": bool(data["is_right"][i]),
            "scaled_focal_length": float(data["scaled_focal_length"][i]),
        }
        if "kp_count" in data:
            det["kp_count"] = int(data["kp_count"][i])
        if "kp_mean_conf" in data:
            det["kp_mean_conf"] = float(data["kp_mean_conf"][i])

        if frame_name not in frames:
            frames[frame_name] = []
        frames[frame_name].append(det)

    return frames


def _load_hamer_params_npz(params_dir):
    if not params_dir.exists():
        return {}

    npz_files = sorted(params_dir.glob("*.npz"))
    if not npz_files:
        return {}

    frames = {}
    for f in npz_files:
        match = re.match(r"^(.+)_(\d+)$", f.stem)
        frame_name = match.group(1) if match else f.stem

        data = dict(np.load(f, allow_pickle=True))
        if frame_name not in frames:
            frames[frame_name] = []
        frames[frame_name].append(data)

    return frames


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge(gvhmr_path: Path | str, hamer_params_dir: Path | str | None, output_path: Path | str, coord_system: str = "global",
          flame_result: Path | str | None = None, gaze_blink_result: Path | str | None = None) -> dict[str, np.ndarray]:
    """Merge GVHMR body + HaMeR hands + FLAME face + gaze/blink into smplx_params.npz.

    Writes a compressed .npz to output_path. See README.md for full key listing.

    Returns:
        dict with all saved keys. Notable keys and shapes (N = num_frames):
        - body_pose (N,63), global_orient (N,3), betas (N,10), transl (N,3)
        - left/right_hand_pose (N,45), left/right_hand_valid (N,) bool
        - jaw_pose (N,3), expression (N,100), leye/reye_pose (N,3), face_valid (N,) bool
        - gaze_pitch (N,), gaze_yaw (N,), gaze_valid (N,) bool
        - blink_left (N,), blink_right (N,)
        - body_valid (N,) bool — False where GVHMR depth is unreliable
        - K_fullimg, num_frames, coord_system
    """

    # Load GVHMR body
    gvhmr_data = load_gvhmr(gvhmr_path)
    coord_key = f"smpl_params_{coord_system}"
    if coord_key not in gvhmr_data:
        available = [k for k in gvhmr_data if k.startswith("smpl_params_")]
        raise ValueError(f"No '{coord_key}' in GVHMR results. Available: {available}")

    body = gvhmr_data[coord_key]
    num_frames = body["body_pose"].shape[0]
    print(f"  GVHMR: {num_frames} frames, body_pose {body['body_pose'].shape}")

    # Load HaMeR hands
    if hamer_params_dir is not None:
        hamer_frames = load_hamer_params(hamer_params_dir)
        print(f"  HaMeR: {len(hamer_frames)} frames with hand detections")
    else:
        hamer_frames = {}
        print(f"  HaMeR: skipped (no hand params provided)")

    relaxed_pose = load_mano_mean_hand_pose()
    left_hand_pose = np.tile(relaxed_pose, (num_frames, 1))
    right_hand_pose = np.tile(relaxed_pose, (num_frames, 1))
    left_hand_valid = np.zeros(num_frames, dtype=bool)
    right_hand_valid = np.zeros(num_frames, dtype=bool)

    n_rejected = 0
    for frame_name, detections in hamer_frames.items():
        try:
            frame_idx = int(frame_name) - 1
        except ValueError:
            continue
        if frame_idx < 0 or frame_idx >= num_frames:
            continue

        for det in detections:
            kp_count = int(det.get("kp_count", 21))
            kp_conf = float(det.get("kp_mean_conf", 1.0))
            if kp_count < MIN_KP_COUNT or kp_conf < MIN_KP_CONF:
                n_rejected += 1
                continue

            is_right = bool(det["is_right"])
            hand_pose_aa = rotmat_batch_to_axis_angle(det["hand_pose"])

            if not is_right:
                hand_pose_aa[:, 1] *= -1
                hand_pose_aa[:, 2] *= -1

            hand_pose_flat = hand_pose_aa.flatten()

            if is_right:
                right_hand_pose[frame_idx] = hand_pose_flat
                right_hand_valid[frame_idx] = True
            else:
                left_hand_pose[frame_idx] = hand_pose_flat
                left_hand_valid[frame_idx] = True

    left_count = left_hand_valid.sum()
    right_count = right_hand_valid.sum()
    print(f"  Merged: left {left_count}/{num_frames}, right {right_count}/{num_frames}"
          f" ({n_rejected} rejected)")

    # SLERP gap interpolation
    if left_count > 0:
        left_hand_pose = interpolate_hand_gaps(left_hand_pose, left_hand_valid, relaxed_pose)
    if right_count > 0:
        right_hand_pose = interpolate_hand_gaps(right_hand_pose, right_hand_valid, relaxed_pose)

    # Assemble SMPL-X parameters
    result = {
        "body_pose": body["body_pose"].copy(),
        "global_orient": body["global_orient"],
        "betas": body["betas"],
        "transl": body["transl"],
        "left_hand_pose": left_hand_pose,
        "right_hand_pose": right_hand_pose,
        "left_hand_valid": left_hand_valid,
        "right_hand_valid": right_hand_valid,
        "jaw_pose": np.zeros((num_frames, 3), dtype=np.float32),
        "leye_pose": np.zeros((num_frames, 3), dtype=np.float32),
        "reye_pose": np.zeros((num_frames, 3), dtype=np.float32),
        "expression": np.zeros((num_frames, 100), dtype=np.float32),
        "face_valid": np.zeros(num_frames, dtype=bool),
        "gaze_pitch": np.zeros(num_frames, dtype=np.float32),
        "gaze_yaw": np.zeros(num_frames, dtype=np.float32),
        "gaze_valid": np.zeros(num_frames, dtype=bool),
        "blink_left": np.zeros(num_frames, dtype=np.float32),
        "blink_right": np.zeros(num_frames, dtype=np.float32),
        "num_frames": num_frames,
        "coord_system": coord_system,
    }

    # Save the other coord system's orient+transl for rendering
    other_coord = "incam" if coord_system == "global" else "global"
    other_key = f"smpl_params_{other_coord}"
    if other_key in gvhmr_data:
        other_body = gvhmr_data[other_key]
        result[f"global_orient_{other_coord}"] = other_body["global_orient"]
        result[f"transl_{other_coord}"] = other_body["transl"]

    # Merge FLAME face params
    if flame_result:
        flame = np.load(flame_result, allow_pickle=True)
        flame_expr = flame["expr"]
        flame_jaw = flame["jaw_pose"]
        flame_eyes = flame["eyes_pose"]
        timestep_ids = flame["timestep_id"]
        if flame_jaw.ndim == 0:
            flame_jaw = np.stack([j.flatten() for j in flame_jaw.item()])
        if flame_expr.ndim == 0:
            flame_expr = np.stack([e.flatten() for e in flame_expr.item()])
        if flame_eyes.ndim == 0:
            flame_eyes = np.stack([e.flatten() for e in flame_eyes.item()])

        face_count = 0
        for i, t in enumerate(timestep_ids):
            t = int(t)
            if 0 <= t < num_frames:
                result["jaw_pose"][t] = flame_jaw[i].flatten()[:3]
                expr_i = flame_expr[i].flatten()
                result["expression"][t, :min(len(expr_i), 100)] = expr_i[:100]
                eyes_i = flame_eyes[i].flatten()
                if len(eyes_i) >= 6:
                    result["leye_pose"][t] = eyes_i[:3]
                    result["reye_pose"][t] = eyes_i[3:6]
                result["face_valid"][t] = True
                face_count += 1

        print(f"  FLAME: {face_count}/{num_frames} frames with face data")

    # Merge gaze + blink
    if gaze_blink_result:
        gb = np.load(gaze_blink_result, allow_pickle=True)
        gb_count = 0
        for i, t in enumerate(gb["timestep_id"]):
            t = int(t)
            if 0 <= t < num_frames:
                result["gaze_pitch"][t] = gb["gaze_pitch"][i]
                result["gaze_yaw"][t] = gb["gaze_yaw"][i]
                result["gaze_valid"][t] = True
                result["blink_left"][t] = gb["blink_left"][i]
                result["blink_right"][t] = gb["blink_right"][i]
                gb_count += 1
        print(f"  Gaze+Blink: {gb_count}/{num_frames} frames")

    if "K_fullimg" in gvhmr_data:
        result["K_fullimg"] = gvhmr_data["K_fullimg"]

    # Depth validation: flag frames where GVHMR depth is unreliable
    # Uses sz/z ratio (bbx_size / estimated_depth) — should be ~constant for a
    # fixed camera. Outliers indicate GVHMR depth failure.
    body_valid = np.ones(num_frames, dtype=bool)
    incam_key = "smpl_params_incam" if "smpl_params_incam" in gvhmr_data else coord_key
    transl_z = gvhmr_data[incam_key]["transl"][:, 2]

    bbx_path = Path(gvhmr_path).parent / "preprocess" / "bbx.pt"
    if bbx_path.exists():
        import torch
        bbx_data = torch.load(str(bbx_path), map_location="cpu", weights_only=False)
        bbx_sz = bbx_data["bbx_xys"][:num_frames, 2].numpy()
        sz_over_z = bbx_sz / np.clip(transl_z, 0.1, None)
        med = np.median(sz_over_z)
        std = np.std(sz_over_z)
        invalid = np.abs(sz_over_z - med) > 3 * std
        # Also flag implausibly far frames
        z_med = np.median(transl_z)
        invalid |= transl_z > 2 * z_med
        body_valid[invalid] = False
        n_invalid = invalid.sum()
        if n_invalid > 0:
            idx = np.where(invalid)[0]
            print(f"  [Depth] Flagged {n_invalid}/{num_frames} frames as body_valid=False "
                  f"(sz/z outliers or z>{2*z_med:.1f}m, range {idx[0]}-{idx[-1]})")
    else:
        # Fallback without bbx
        z_med = np.median(transl_z)
        invalid = transl_z > 2 * z_med
        if invalid.sum() > 0:
            body_valid[invalid] = False
            print(f"  [Depth] Flagged {invalid.sum()}/{num_frames} frames (z>{2*z_med:.1f}m, no bbx)")

    result["body_valid"] = body_valid

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **result)
    print(f"  Saved: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge body + hands + face -> SMPL-X")
    parser.add_argument("--gvhmr_result", type=str, required=True)
    parser.add_argument("--hamer_result", type=str, required=False, default=None)
    parser.add_argument("--flame_result", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--coord", type=str, default="global", choices=["global", "incam"])
    parser.add_argument("--gaze_blink_result", type=str, default=None)
    args = parser.parse_args()

    print(f"=== Merging Body + Hands + Face -> SMPL-X ===")
    print(f"GVHMR:  {args.gvhmr_result}")
    print(f"HaMeR:  {args.hamer_result}")
    print(f"FLAME:  {args.flame_result or 'None'}")
    print(f"Gaze:   {args.gaze_blink_result or 'None'}")
    print(f"Output: {args.output}")
    print(f"Coords: {args.coord}")
    print()

    merge(args.gvhmr_result, args.hamer_result, args.output, args.coord,
          flame_result=args.flame_result,
          gaze_blink_result=args.gaze_blink_result)


if __name__ == "__main__":
    main()
