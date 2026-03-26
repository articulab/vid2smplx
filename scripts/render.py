#!/usr/bin/env python3
"""Unified renderer for vid2smplx: body, hands, face, gaze overlays.

Mode:
  --clip_dir + --layers: render layers from clip directory
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure GVHMR is on Python path so hmr4d imports work from any cwd
_repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo / "GVHMR"))

import cv2
import numpy as np
import smplx
import torch
from tqdm import tqdm

from hmr4d.utils.video_io_utils import get_video_lwh, get_writer, get_video_reader, merge_videos_horizontal
from hmr4d.utils.vis.renderer import get_global_cameras_static, get_ground_params_from_points
from hmr4d.utils.geo.hmr_cam import create_camera_sensor
from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from hmr4d.utils.net_utils import to_cuda
from einops import einsum

from utils import (
    get_renderer_class, ensure_max_resolution, load_mano_faces,
    load_mano_joint_regressor, smplx_forward_chunked, _get_repo_dir,
    HAMER_FOCAL_LENGTH, HAMER_IMAGE_SIZE,
    CRF, RENDER_SCALE, MAX_RESOLUTION,
)
from hand_utils import (
    load_hand_meshes, compute_incam_to_global_transform, place_hands_dc,
)
# IK is handled by ik_hands.py — render loads results from smplx_params.npz

Renderer = get_renderer_class()


# ---------------------------------------------------------------------------
# Gaze arrow drawing
# ---------------------------------------------------------------------------

def draw_gaze_arrow(img, nose_2d, pitch, yaw, arrow_length=150,
                    color=(0, 255, 255), thickness=3):
    """Draw a gaze direction arrow on an image."""
    x, y = int(nose_2d[0]), int(nose_2d[1])
    dx = -arrow_length * np.sin(yaw) * np.cos(pitch)
    dy = -arrow_length * np.sin(pitch)
    end_x, end_y = int(x + dx), int(y + dy)
    cv2.arrowedLine(img, (x, y), (end_x, end_y), color, thickness, tipLength=0.3)
    return img


# ---------------------------------------------------------------------------
# EMICA ortho-to-perspective conversion (for face rendering)
# ---------------------------------------------------------------------------

def emica_ortho_to_cam_t_full(s, tx, ty, bbox, focal_length, img_w, img_h):
    """Convert EMICA orthographic cam to perspective cam_t_full."""
    x1, y1, x2, y2 = bbox
    box_cx = (x1 + x2) / 2.0
    box_cy = (y1 + y2) / 2.0
    box_size = max(x2 - x1, y2 - y1)

    bs = box_size * s + 1e-9
    tz = 2 * focal_length / bs
    tx_full = (2 * (box_cx - img_w / 2.0) / bs) + tx / s
    ty_full = (2 * (box_cy - img_h / 2.0) / bs) + ty / s

    return np.array([tx_full, ty_full, tz], dtype=np.float32)


# ---------------------------------------------------------------------------
# Incam renderer
# ---------------------------------------------------------------------------

def render_incam(video_path: str, verts: torch.Tensor, K: torch.Tensor, output_path: str, faces: torch.Tensor,
                 hand_data: dict[int, list[dict]] | None = None, mano_faces_right: torch.Tensor | None = None, mano_faces_left: torch.Tensor | None = None,
                 gaze_data: dict[str, np.ndarray] | None = None, joints_incam: torch.Tensor | None = None) -> None:
    """Render body (+ optional MANO hands + gaze arrows) overlaid on video."""
    length, width, height = get_video_lwh(video_path)
    body_renderer = Renderer(width, height, device="cuda", faces=faces, K=K, render_scale=RENDER_SCALE)

    hand_renderer_right = None
    hand_renderer_left = None
    if hand_data and mano_faces_right is not None:
        hand_renderer_right = Renderer(width, height, device="cuda", faces=mano_faces_right, K=K, render_scale=RENDER_SCALE)
        hand_renderer_left = Renderer(width, height, device="cuda", faces=mano_faces_left, K=K, render_scale=RENDER_SCALE)

    color_right = [0.9, 0.7, 0.5]
    color_left = [0.5, 0.7, 0.9]

    nose_2d_per_frame = {}
    if gaze_data is not None and joints_incam is not None:
        fx, fy = K[0, 0].item(), K[1, 1].item()
        cx, cy = K[0, 2].item(), K[1, 2].item()
        nose_idx = 55
        for t in gaze_data["timestep_id"]:
            t = int(t)
            if t < joints_incam.shape[0]:
                j3d = joints_incam[t, nose_idx]
                if j3d[2] > 0:
                    px = (j3d[0] / j3d[2]) * fx + cx
                    py = (j3d[1] / j3d[2]) * fy + cy
                    nose_2d_per_frame[t] = (float(px), float(py))

    reader = get_video_reader(video_path)
    writer = get_writer(output_path, fps=30, crf=CRF)
    for i, img_raw in tqdm(enumerate(reader), total=length, desc="Rendering Incam"):
        if i >= len(verts):
            break
        img = body_renderer.render_mesh(verts[i].cuda(), img_raw, [0.8, 0.8, 0.8])

        if hand_data and i in hand_data:
            for hd in hand_data[i]:
                hv = hd["verts"].cuda()
                if hd["is_right"]:
                    img = hand_renderer_right.render_mesh(hv, img, color_right)
                else:
                    img = hand_renderer_left.render_mesh(hv, img, color_left)

        if gaze_data is not None and i in nose_2d_per_frame:
            idx = np.searchsorted(gaze_data["timestep_id"], i)
            if idx < len(gaze_data["timestep_id"]) and int(gaze_data["timestep_id"][idx]) == i:
                pitch = float(gaze_data["gaze_pitch"][idx])
                yaw = float(gaze_data["gaze_yaw"][idx])
                if abs(pitch) > 1e-6 or abs(yaw) > 1e-6:
                    img = draw_gaze_arrow(img, nose_2d_per_frame[i], pitch, yaw)

        writer.write_frame(img)
    writer.close()
    reader.close()
    print(f"  [Incam] Saved: {output_path}")


# ---------------------------------------------------------------------------
# Global scene preparation
# ---------------------------------------------------------------------------

def prepare_global_scene(verts: torch.Tensor, hand_data: dict[int, list[dict]] | None, body_model_dir: Path | str | None = None) -> tuple[torch.Tensor, dict[int, list[dict]] | None, torch.Tensor]:
    """Precompute scene normalization for global views."""
    if body_model_dir is None:
        body_model_dir = Path("hmr4d/utils/body_model")
    body_model_dir = Path(body_model_dir)
    J_regressor = torch.load(body_model_dir / "smpl_neutral_J_regressor.pt").cuda()
    smplx2smpl = torch.load(body_model_dir / "smplx2smpl_sparse.pt").cuda()
    smpl_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in verts.cuda()])
    joints_for_cam = einsum(J_regressor, smpl_verts, "j v, l v i -> l j i")

    offset = joints_for_cam[0, 0].clone()
    offset[1] = verts.cuda()[:, :, 1].min()

    def move_to_start_point_face_z(v):
        v = v.clone() - offset
        smpl_v = torch.stack([torch.matmul(smplx2smpl, vi) for vi in v])
        j = einsum(J_regressor, smpl_v, "j v, l v i -> l j i")
        T_ay2ayfz = compute_T_ayfz2ay(j[[0]], inverse=True)
        v = apply_T_on_points(v, T_ay2ayfz)
        return v, T_ay2ayfz

    verts_scene, T_ay2ayfz = move_to_start_point_face_z(verts.cuda())

    hands_scene = None
    if hand_data:
        hands_scene = {}
        for frame_idx, dets in hand_data.items():
            hands_scene[frame_idx] = []
            for hd in dets:
                hv = hd["verts"].cuda() - offset
                hv_h = torch.cat([hv, torch.ones(hv.shape[0], 1, device=hv.device)], dim=1)
                hv_transformed = (T_ay2ayfz[0] @ hv_h.T).T[:, :3]
                hands_scene[frame_idx].append({
                    "verts": hv_transformed,
                    "is_right": hd["is_right"],
                })

    smpl_verts_scene = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in verts_scene])
    joints_scene = einsum(J_regressor, smpl_verts_scene, "j v, l v i -> l j i")

    return verts_scene, hands_scene, joints_scene


# ---------------------------------------------------------------------------
# Global triview renderer
# ---------------------------------------------------------------------------

def render_global_triview(verts_scene: torch.Tensor, output_dir: Path | str, faces: torch.Tensor, width: int, height: int, joints_scene: torch.Tensor,
                          hands_scene: dict[int, list[dict]] | None = None, mano_faces_right: torch.Tensor | None = None, mano_faces_left: torch.Tensor | None = None,
                          view_configs: list[tuple[str, int, str]] | None = None, label_prefix: str = "") -> list[str]:
    """Render 3 global views (right/front/left)."""
    if view_configs is None:
        view_configs = [
            ("right.mp4", 45, f" {label_prefix}right"),
            ("front.mp4", 0, f" {label_prefix}front"),
            ("left.mp4", -45, f" {label_prefix}left"),
        ]

    output_paths = [str(Path(output_dir) / vc[0]) for vc in view_configs]

    _, _, K = create_camera_sensor(width, height, 24)
    renderer = Renderer(width, height, device="cuda", faces=faces, K=K, render_scale=RENDER_SCALE)
    scale, cx_g, cz = get_ground_params_from_points(joints_scene[:, 0], verts_scene)
    renderer.set_ground(scale * 1.5, cx_g, cz)

    body_color = torch.ones(3).float().cuda() * 0.8
    color_right_t = torch.tensor([0.9, 0.7, 0.5]).float().cuda()
    color_left_t = torch.tensor([0.5, 0.7, 0.9]).float().cuda()

    cam_data = []
    for _, vec_rot, _ in view_configs:
        R, T, lights = get_global_cameras_static(
            verts_scene.cpu(), beta=2.0, cam_height_degree=20,
            target_center_height=1.0, vec_rot=vec_rot,
        )
        cam_data.append((R, T, lights))

    n_views = len(view_configs)
    length = len(verts_scene)
    writers = [get_writer(p, fps=30, crf=CRF) for p in output_paths]

    use_batched = hasattr(renderer, 'render_triview')

    if use_batched:
        # get_global_cameras_static returns PT3D convention (R stored as .mT, y-up, z-away).
        # render_triview expects column-vector R in OpenCV convention (y-down, z-towards).
        # Convert: undo .mT storage, then flip Y and Z axes.
        flip_yz = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]],
                               dtype=torch.float32, device="cuda")
        R_views = torch.stack([flip_yz @ cam_data[vi][0][0].to("cuda").view(3, 3).mT for vi in range(n_views)])
        T_views = torch.stack([flip_yz @ cam_data[vi][1][0].to("cuda").view(3) for vi in range(n_views)])

        gv, gf, gc = renderer.ground_geometry
        gv = gv.to("cuda").float()
        gf = gf.to("cuda").to(torch.int32)
        gc = gc.to("cuda").float()[..., :3]

        body_faces = faces.to("cuda").to(torch.int32)
        mano_fr = mano_faces_right.to("cuda").to(torch.int32) if mano_faces_right is not None else None
        mano_fl = mano_faces_left.to("cuda").to(torch.int32) if mano_faces_left is not None else None

        for i in tqdm(range(length), desc=f"Rendering Global{label_prefix} ({n_views} views)"):
            all_verts, all_faces, all_colors = [], [], []
            offset = 0

            bv = verts_scene[i].to("cuda").float()
            nv = bv.shape[0]
            all_verts.append(bv)
            all_faces.append(body_faces + offset)
            all_colors.append(body_color.unsqueeze(0).expand(nv, -1))
            offset += nv

            if hands_scene and i in hands_scene:
                for hd in hands_scene[i]:
                    hv = hd["verts"].to("cuda").float()
                    nh = hv.shape[0]
                    h_faces = mano_fr if hd["is_right"] else mano_fl
                    h_color = color_right_t if hd["is_right"] else color_left_t
                    all_verts.append(hv)
                    all_faces.append(h_faces + offset)
                    all_colors.append(h_color.unsqueeze(0).expand(nh, -1))
                    offset += nh

            all_verts.append(gv)
            all_faces.append(gf + offset)
            all_colors.append(gc)

            images = renderer.render_triview(
                torch.cat(all_verts), torch.cat(all_faces),
                torch.cat(all_colors), R_views, T_views,
            )
            for vi in range(n_views):
                writers[vi].write_frame(images[vi])
    else:
        hand_renderer_right = None
        hand_renderer_left = None
        if hands_scene and mano_faces_right is not None:
            hand_renderer_right = Renderer(width, height, device="cuda", faces=mano_faces_right.clone().to("cuda"), K=K, render_scale=RENDER_SCALE)
            hand_renderer_left = Renderer(width, height, device="cuda", faces=mano_faces_left.clone().to("cuda"), K=K, render_scale=RENDER_SCALE)

        for i in tqdm(range(length), desc=f"Rendering Global{label_prefix} ({n_views} views)"):
            for vi in range(n_views):
                R, T, lights = cam_data[vi]
                cameras = renderer.create_camera(R[i], T[i])
                img = renderer.render_with_ground(verts_scene[[i]], body_color[None], cameras, lights)

                if hands_scene and i in hands_scene:
                    cam_R = R[i].to("cuda").view(3, 3)
                    cam_T = T[i].to("cuda").view(3)
                    for hd in hands_scene[i]:
                        hv = hd["verts"]
                        color = color_right_t if hd["is_right"] else color_left_t
                        hv_cam = (cam_R @ hv.T).T + cam_T
                        if hd["is_right"]:
                            img = hand_renderer_right.render_mesh(hv_cam, img, color.tolist())
                        else:
                            img = hand_renderer_left.render_mesh(hv_cam, img, color.tolist())

                writers[vi].write_frame(img)

    for vi, (_, _, label) in enumerate(view_configs):
        writers[vi].close()
        print(f"  [Global{label}] Saved: {output_paths[vi]}")

    return output_paths


# ---------------------------------------------------------------------------
# Standalone face renderer
# ---------------------------------------------------------------------------

def render_face_incam(video_path: Path | str, emica_result_path: Path | str, gvhmr_result_path: Path | str, output_path: Path | str) -> None:
    """Render FLAME face mesh overlaid on video."""
    video_path = ensure_max_resolution(video_path)

    length, width, height = get_video_lwh(video_path)
    print(f"Video: {length} frames, {width}x{height}")

    emica = dict(np.load(emica_result_path, allow_pickle=True))
    verts_all = emica["verts"]
    cam_all = emica["cam"]
    bboxes = emica["face_bbox"]
    timestep_ids = emica["timestep_id"]
    faces = emica["faces"]

    pred = torch.load(gvhmr_result_path, map_location="cpu", weights_only=False)
    K = pred["K_fullimg"][0]
    if isinstance(K, torch.Tensor):
        K = K.numpy()
    else:
        K = np.array(K, dtype=np.float32)
    focal_length = float(K[0, 0])

    frame_to_emica = {int(t): i for i, t in enumerate(timestep_ids)}

    flame_faces = torch.from_numpy(faces.astype(np.int64))
    renderer = Renderer(width, height, device="cuda", faces=flame_faces,
                        K=torch.from_numpy(K).float(), render_scale=RENDER_SCALE)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reader = get_video_reader(video_path)
    writer = get_writer(str(output_path), fps=30, crf=CRF)

    n_rendered = 0
    for i, img_raw in tqdm(enumerate(reader), total=length, desc="Rendering face incam"):
        if i >= length:
            break

        if i not in frame_to_emica:
            writer.write_frame(img_raw)
            continue

        ei = frame_to_emica[i]
        verts = verts_all[ei]
        s, tx, ty = cam_all[ei]
        bbox = bboxes[ei]

        cam_t = emica_ortho_to_cam_t_full(s, tx, ty, bbox, focal_length, width, height)

        verts_cam = verts.copy()
        verts_cam[:, 1] *= -1
        verts_cam[:, 2] *= -1
        verts_incam = verts_cam + cam_t

        verts_t = torch.from_numpy(verts_incam.astype(np.float32)).cuda()
        img = renderer.render_mesh(verts_t, img_raw, [0.8, 0.8, 0.8])
        writer.write_frame(img)
        n_rendered += 1

    writer.close()
    reader.close()
    print(f"Rendered {n_rendered} frames with face mesh -> {output_path}")


# ---------------------------------------------------------------------------
# Standalone hands renderer
# ---------------------------------------------------------------------------

def render_hands_incam(video_path: Path | str, mano_params_path: Path | str, output_path: Path | str) -> None:
    """Render standalone MANO hand meshes overlaid on video."""
    video_path = ensure_max_resolution(video_path)

    length, width, height = get_video_lwh(video_path)
    print(f"Video: {length} frames, {width}x{height}")

    by_frame = load_hand_meshes(mano_params_path, length)
    default_focal = HAMER_FOCAL_LENGTH / HAMER_IMAGE_SIZE * max(width, height)
    first_det = next((dets[0] for dets in by_frame.values() if dets), None)
    focal_length = (first_det["scaled_focal_length"]
                    if first_det and first_det["scaled_focal_length"] is not None
                    else default_focal)
    print(f"Focal length: {focal_length:.1f}")

    K = torch.zeros(3, 3)
    K[0, 0] = focal_length
    K[1, 1] = focal_length
    K[0, 2] = width / 2.0
    K[1, 2] = height / 2.0
    K[2, 2] = 1.0

    n_detections = sum(len(v) for v in by_frame.values())
    print(f"Hand detections: {n_detections} across {len(by_frame)} frames")
    if n_detections == 0:
        print("No hand meshes found. Nothing to render.")
        return

    mano_faces_np = load_mano_faces()
    if mano_faces_np is None:
        return

    faces_right = torch.from_numpy(mano_faces_np.astype(np.int64))
    faces_left = torch.from_numpy(mano_faces_np[:, ::-1].copy().astype(np.int64))

    renderer_right = Renderer(width, height, device="cuda", faces=faces_right, K=K, render_scale=RENDER_SCALE)
    renderer_left = Renderer(width, height, device="cuda", faces=faces_left, K=K, render_scale=RENDER_SCALE)

    color_right = [0.9, 0.7, 0.5]
    color_left = [0.5, 0.7, 0.9]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reader = get_video_reader(video_path)
    writer = get_writer(str(output_path), fps=30, crf=CRF)

    for i, img_raw in tqdm(enumerate(reader), total=length, desc="Rendering hands incam"):
        if i >= length:
            break

        detections = by_frame.get(i, [])
        if not detections:
            writer.write_frame(img_raw)
            continue

        img = img_raw
        for det in detections:
            verts = det["vertices"].copy()
            cam_t = det["cam_t_full"]
            is_right = det["is_right"]

            verts[:, 0] = (2 * int(is_right) - 1) * verts[:, 0]
            verts_cam = verts + cam_t[None, :]
            verts_t = torch.from_numpy(verts_cam).float().cuda()

            if is_right:
                img = renderer_right.render_mesh(verts_t, img, color_right)
            else:
                img = renderer_left.render_mesh(verts_t, img, color_left)

        writer.write_frame(img)

    writer.close()
    reader.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Shared helpers for layer rendering
# ---------------------------------------------------------------------------

def _resolve_smplx_dir(smplx_dir):
    """Find SMPL-X model directory."""
    if smplx_dir is not None:
        smplx_dir = str(Path(smplx_dir).resolve())
        if (Path(smplx_dir) / "SMPLX_NEUTRAL.npz").exists():
            smplx_dir = str(Path(smplx_dir).parent)
        return smplx_dir
    for candidate in [Path(".."), Path("inputs/checkpoints/body_models")]:
        if (candidate / "smplx" / "SMPLX_NEUTRAL.npz").exists():
            return str(candidate.resolve())
    return None



def _load_gvhmr_and_model(gvhmr_result, smplx_dir, smplx_params_npz=None):
    """Load GVHMR prediction and create SMPL-X model. Returns (pred, model, faces, K, length)."""
    pred = torch.load(gvhmr_result, map_location="cpu", weights_only=False)
    incam = pred["smpl_params_incam"]
    length = len(incam["body_pose"])

    model = smplx.create(
        smplx_dir, model_type="smplx", gender="neutral",
        use_pca=False, flat_hand_mean=True, num_betas=10,
        num_expression_coeffs=100, batch_size=1,
    ).cuda()
    faces = torch.from_numpy(model.faces.astype(np.int64))

    K = pred["K_fullimg"][0]
    K = K.clone().float() if isinstance(K, torch.Tensor) else torch.tensor(K).float()

    # Load face params from smplx_params.npz if available
    face_params = {}
    if smplx_params_npz and Path(smplx_params_npz).exists():
        data = dict(np.load(smplx_params_npz, allow_pickle=True))
        for key in ["jaw_pose", "expression", "leye_pose", "reye_pose"]:
            if key in data and np.any(data[key] != 0):
                face_params[key] = torch.from_numpy(data[key][:length]).float()

    return pred, model, faces, K, length, face_params


def _build_gvhmr_params(pred, coord, length, face_params=None):
    """Build SMPL-X forward-pass params from GVHMR prediction (no IK, zero hands).

    Returns:
        dict with keys: body_pose (N,63), global_orient (N,3), betas (N,10),
        transl (N,3), left/right_hand_pose (N,45) as zeros, plus any face_params
        (jaw_pose, expression, leye_pose, reye_pose) if provided.
    """
    body = pred[f"smpl_params_{coord}"]
    params = {
        "body_pose": torch.tensor(body["body_pose"]).float(),
        "global_orient": torch.tensor(body["global_orient"]).float(),
        "betas": torch.tensor(body["betas"]).float(),
        "transl": torch.tensor(body["transl"]).float(),
        "left_hand_pose": torch.zeros(length, 45),
        "right_hand_pose": torch.zeros(length, 45),
    }
    if face_params:
        for key, val in face_params.items():
            params[key] = val[:length]
    return params


def _load_ik_params(smplx_params_npz, length):
    """Load IK-corrected params from smplx_params.npz.

    Returns:
        dict with keys body_pose, global_orient, transl, betas,
        left/right_hand_pose (plus jaw_pose, expression, leye/reye_pose if present),
        all as float tensors of shape (length, D). Prefers _incam variants.
        Returns None if no IK results found (ik_wrist_loss key absent).
    """
    if not smplx_params_npz or not Path(smplx_params_npz).exists():
        return None
    data = dict(np.load(smplx_params_npz, allow_pickle=True))
    if "ik_wrist_loss" not in data:
        return None

    params = {}
    for key in ["body_pose", "betas", "left_hand_pose", "right_hand_pose",
                "global_orient", "transl"]:
        incam_key = key + "_incam"
        if incam_key in data:
            params[key] = torch.from_numpy(data[incam_key][:length]).float()
        elif key in data:
            params[key] = torch.from_numpy(data[key][:length]).float()
    for key in ["jaw_pose", "expression", "leye_pose", "reye_pose"]:
        if key in data and np.any(data[key] != 0):
            params[key] = torch.from_numpy(data[key][:length]).float()
    return params


# ---------------------------------------------------------------------------
# Layer rendering from clip directory
# ---------------------------------------------------------------------------

VALID_LAYERS = {"gvhmr", "hands", "face", "final", "global"}


def render_layers(clip_dir: Path | str, layers: set[str], smplx_dir: Path | str, force: bool = False, video_override: Path | str | None = None) -> None:
    """Render requested layers from saved intermediates in a clip directory.

    Args:
        clip_dir: Path to output/corpus/<clip_name>/
        layers: set of layer names from VALID_LAYERS
            - "gvhmr": raw GVHMR body overlay -> gvhmr_incam.mp4
            - "hands": standalone HaMeR hands overlay -> hands_incam.mp4
            - "face": EMICA FLAME face overlay -> face_incam.mp4
            - "final": body + IK hands + face + gaze -> final_incam.mp4
            - "global": three fixed-camera views -> front.mp4, left.mp4, right.mp4
        smplx_dir: path to directory containing smplx/SMPLX_NEUTRAL.npz
        force: if True, re-render even if output exists
        video_override: if set, use this video instead of the default input video

    Writes MP4s into clip_dir/render/. No return value.
    """
    clip_dir = Path(clip_dir)
    clip_name = clip_dir.name
    render_dir = clip_dir / "render"
    render_dir.mkdir(parents=True, exist_ok=True)

    # Discover available intermediates
    gvhmr_result = clip_dir / "gvhmr" / clip_name / "hmr4d_results.pt"
    video = Path(video_override) if video_override else clip_dir / "gvhmr" / clip_name / "0_input_video.mp4"
    hamer_params = clip_dir / "hamer" / clip_name / "rendered" / "hamer_hands.pt"
    smplx_params_npz = clip_dir / "smplx_params.npz"
    emica_result = clip_dir / "emica" / clip_name / "flame_params.npz"
    gaze_result = clip_dir / "gaze_blink" / clip_name / "gaze_blink.npz"
    if not gaze_result.exists():
        gaze_result = clip_dir / "gaze_blink.npz"

    if not video.exists():
        print(f"ERROR: Video not found: {video}")
        sys.exit(1)
    if not gvhmr_result.exists() and layers & {"gvhmr", "final", "global"}:
        print(f"ERROR: GVHMR result not found: {gvhmr_result}")
        sys.exit(1)

    video_str = ensure_max_resolution(str(video))

    print(f"=== Rendering layers: {', '.join(sorted(layers))} ===")
    print(f"Clip:  {clip_name}")
    print(f"Video: {video_str}")
    print()

    # -- Layer: hands --
    if "hands" in layers:
        out = render_dir / "hands_incam.mp4"
        if out.exists() and not force:
            print(f"[SKIP] {out}")
        elif not hamer_params.exists():
            print("[SKIP] hands — no HaMeR params")
        else:
            print("=== Rendering: hands_incam ===")
            render_hands_incam(video_str, str(hamer_params), str(out))

    # -- Layer: face --
    if "face" in layers:
        out = render_dir / "face_incam.mp4"
        if out.exists() and not force:
            print(f"[SKIP] {out}")
        elif not emica_result.exists():
            print("[SKIP] face — no EMICA params")
        else:
            print("=== Rendering: face_incam ===")
            render_face_incam(video_str, str(emica_result), str(gvhmr_result), str(out))

    # Load shared resources for body layers
    if layers & {"gvhmr", "final", "global"}:
        smplx_npz_str = str(smplx_params_npz) if smplx_params_npz.exists() else None
        pred, model, smplx_faces, K, length, face_params = _load_gvhmr_and_model(
            str(gvhmr_result), smplx_dir, smplx_npz_str,
        )

    # -- Layer: gvhmr (raw body, no IK, no hands) --
    if "gvhmr" in layers:
        out = render_dir / "gvhmr_incam.mp4"
        if out.exists() and not force:
            print(f"[SKIP] {out}")
        else:
            print("=== Rendering: gvhmr_incam (raw body, no IK) ===")
            params = _build_gvhmr_params(pred, "incam", length, face_params)
            verts, joints = smplx_forward_chunked(model, params)
            render_incam(video_str, verts, K, str(out), smplx_faces,
                         joints_incam=joints)

    # -- Layer: final (body + IK hands + face) --
    if "final" in layers:
        out = render_dir / "final_incam.mp4"
        if out.exists() and not force:
            print(f"[SKIP] {out}")
        else:
            print("=== Rendering: final_incam (body + IK hands + face) ===")
            ik_params = _load_ik_params(str(smplx_params_npz), length)
            if ik_params:
                print("  Using IK-corrected params")
                verts, joints = smplx_forward_chunked(model, ik_params)
            else:
                print("  No IK found — using raw GVHMR params")
                params = _build_gvhmr_params(pred, "incam", length, face_params)
                verts, joints = smplx_forward_chunked(model, params)

            # Load gaze
            gaze_data = None
            if gaze_result.exists():
                gaze_data = dict(np.load(str(gaze_result), allow_pickle=True))

            render_incam(video_str, verts, K, str(out), smplx_faces,
                         gaze_data=gaze_data, joints_incam=joints)

    # -- Layer: global --
    if "global" in layers:
        out_front = render_dir / "front.mp4"
        if out_front.exists() and not force:
            print(f"[SKIP] globals — already exist")
        elif "smpl_params_global" not in pred:
            print("[SKIP] global — no global params in GVHMR result")
        else:
            print("=== Rendering: global triview ===")
            ik_params = _load_ik_params(str(smplx_params_npz), length)
            if ik_params:
                global_params = _build_gvhmr_params(pred, "global", length, face_params)
                # Use IK body_pose (arms+hands) with global orient+transl
                global_params["body_pose"] = ik_params["body_pose"]
                global_params["left_hand_pose"] = ik_params["left_hand_pose"]
                global_params["right_hand_pose"] = ik_params["right_hand_pose"]
                verts_g, _ = smplx_forward_chunked(model, global_params)
            else:
                global_params = _build_gvhmr_params(pred, "global", length, face_params)
                verts_g, _ = smplx_forward_chunked(model, global_params)

            length_w, width, height = get_video_lwh(video_str)
            body_model_dir = _get_repo_dir() / "GVHMR" / "hmr4d" / "utils" / "body_model"
            verts_scene, _, joints_scene = prepare_global_scene(verts_g, None, body_model_dir=body_model_dir)
            render_global_triview(verts_scene, str(render_dir), smplx_faces,
                                  width, height, joints_scene)

    # Summary
    print(f"\n=== Renders ===")
    for f in sorted(render_dir.glob("*.mp4")):
        print(f"  {f.name}: {f.stat().st_size / 1e6:.1f}MB")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render vid2smplx results",
        epilog="""
Usage:
  python render.py --clip_dir output/corpus/V00_... --layers gvhmr,hands,face,final
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--clip_dir", type=str, required=True,
                        help="Clip directory (e.g. output/corpus/V00_...)")
    parser.add_argument("--layers", type=str, default=None,
                        help=f"Comma-separated layers to render: {','.join(sorted(VALID_LAYERS))}")
    parser.add_argument("--force", action="store_true",
                        help="Re-render even if output exists")
    parser.add_argument("--video", type=str, default=None,
                        help="Video path override")
    parser.add_argument("--smplx_dir", type=str, default=None)

    args = parser.parse_args()

    if args.clip_dir:
        smplx_dir = _resolve_smplx_dir(args.smplx_dir)
        if smplx_dir is None:
            print("ERROR: Cannot find SMPL-X model files. Use --smplx_dir")
            sys.exit(1)

        if args.layers:
            layers = set(args.layers.split(","))
            bad = layers - VALID_LAYERS
            if bad:
                print(f"ERROR: Unknown layers: {bad}. Valid: {VALID_LAYERS}")
                sys.exit(1)
        else:
            layers = {"gvhmr", "hands", "face", "final"}

        render_layers(args.clip_dir, layers, smplx_dir, force=args.force,
                      video_override=args.video)


if __name__ == "__main__":
    main()
