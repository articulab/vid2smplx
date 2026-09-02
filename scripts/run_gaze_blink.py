#!/usr/bin/env python3
"""Run L2CS-Net gaze estimation + MediaPipe EAR blink detection on a video.

Batched processing: reads frames from video, crops faces, runs L2CS in batches.

Output: <out_folder>/<video_name>/gaze_blink.npz

Usage:
    python scripts/run_gaze_blink.py --video input.mp4 --out_folder output/gaze_blink
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from utils import LEFT_EYE_EAR, RIGHT_EYE_EAR, crop_face


# ---------------------------------------------------------------------------
# Blink: Eye Aspect Ratio via MediaPipe
# ---------------------------------------------------------------------------

def landmarks_to_px(landmarks: object, sx: float, sy: float,
                    ox: float = 0.0, oy: float = 0.0) -> np.ndarray:
    """MediaPipe normalized landmarks -> (N, 2) full-frame pixel coords.

    (sx, sy) is the size of whatever image was fed to FaceMesh and (ox, oy) its
    origin in the full frame, so ROI and full-frame results share one space.
    """
    return np.array([[l.x * sx + ox, l.y * sy + oy] for l in landmarks.landmark],
                    dtype=np.float32)


def compute_ear(pts_px: np.ndarray, indices: list[int]) -> float:
    """Compute Eye Aspect Ratio from 6 landmarks given in full-frame pixels.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    pts = pts_px[indices]

    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    horiz = np.linalg.norm(pts[0] - pts[3])

    if horiz < 1e-6:
        return 0.0
    return float((v1 + v2) / (2.0 * horiz))


# FaceMesh cost scales with input pixels. EMICA has already localized the face,
# so feed it a small ROI instead of the full 1080x1920 frame.
ROI_SCALE = 1.8    # context around the bbox — FaceMesh needs margin to lock on
ROI_MIN_PX = 192


def roi_from_bbox(bbox, w: int, h: int):
    """Square, frame-clamped ROI around a face bbox. None if degenerate."""
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = max(max(x1 - x0, y1 - y0) * ROI_SCALE, ROI_MIN_PX)
    rx0, ry0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    rx1, ry1 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    if rx1 - rx0 < 16 or ry1 - ry0 < 16:
        return None
    return rx0, ry0, rx1, ry1


# ---------------------------------------------------------------------------
# Gaze: L2CS-Net (batched)
# ---------------------------------------------------------------------------

GAZE_INPUT_SIZE = 448
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NUM_BINS = 90
BIN_WIDTH = 4


def load_gaze_model(corpus_dir: Path | str | None = None) -> tuple[torch.nn.Module, torch.device]:
    """Load L2CS-Net ResNet50 model directly."""
    from l2cs import getArch

    search_paths = [Path("models/L2CSNet_gaze360.pkl")]
    if corpus_dir:
        search_paths.insert(0, Path(corpus_dir) / "models" / "L2CSNet_gaze360.pkl")

    weights_path = None
    for p in search_paths:
        if p.exists():
            weights_path = p
            break

    if weights_path is None:
        raise FileNotFoundError(
            "L2CS-Net weights not found. Download L2CSNet_gaze360.pkl from:\n"
            "  https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd\n"
            f"and place it at: {search_paths[0]}"
        )

    print(f"  L2CS weights: {weights_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = getArch("ResNet50", NUM_BINS)
    state_dict = torch.load(str(weights_path), map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device


def decode_gaze(yaw_logits: torch.Tensor, pitch_logits: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Convert L2CS bin logits to continuous angles (radians)."""
    softmax = torch.nn.functional.softmax
    idx_tensor = torch.arange(NUM_BINS, dtype=torch.float32, device=yaw_logits.device)

    yaw_deg = torch.sum(softmax(yaw_logits, dim=1) * idx_tensor, dim=1) * BIN_WIDTH - 180
    pitch_deg = torch.sum(softmax(pitch_logits, dim=1) * idx_tensor, dim=1) * BIN_WIDTH - 180

    return (pitch_deg * np.pi / 180.0).cpu().numpy(), (yaw_deg * np.pi / 180.0).cpu().numpy()


# ImageNet normalization constants as tensors (for direct CHW float32 normalize)
_IMAGENET_MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_IMAGENET_STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)


def _run_gaze_batch(batch, frame_ids, model, device, pitches, yaws):
    """Run L2CS inference on a batch of pre-normalized crop tensors."""
    batch_tensor = torch.stack(batch).to(device)
    with torch.no_grad():
        yaw_logits, pitch_logits = model(batch_tensor)
    p, y = decode_gaze(yaw_logits, pitch_logits)
    for i, fid in enumerate(frame_ids):
        pitches[fid] = p[i]
        yaws[fid] = y[i]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run gaze (L2CS-Net) + blink (MediaPipe EAR) on video"
    )
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--out_folder", type=str, required=True)
    parser.add_argument("--emica_cache", type=str, default=None,
                        help="Path to EMICA _detection_cache.npz (reuse face bboxes)")
    parser.add_argument("--corpus_dir", type=str, default=None)
    parser.add_argument("--video_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    t_start = time.time()

    video_path = Path(args.video).resolve()
    video_name = args.video_name or video_path.stem
    out_dir = Path(args.out_folder).resolve() / video_name
    out_dir.mkdir(parents=True, exist_ok=True)

    final_npz = out_dir / "gaze_blink.npz"
    if final_npz.exists():
        print(f"  [SKIP] Already exists: {final_npz}")
        return

    print(f"=== Gaze + Blink Estimation (batched) ===")
    print(f"Video:  {video_path}")
    print(f"Output: {out_dir}")

    # Load EMICA detection cache if available
    emica_bbox_map = {}
    if args.emica_cache and Path(args.emica_cache).exists():
        cache = np.load(args.emica_cache, allow_pickle=True)
        for i, idx in enumerate(cache["valid_indices"]):
            emica_bbox_map[int(idx)] = cache["bboxes"][i]
        print(f"  Loaded EMICA cache: {len(emica_bbox_map)} face bboxes")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Total frames: {total_frames}")

    # Initialize MediaPipe
    import mediapipe as mp
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    mp_face_det = None
    if not emica_bbox_map:
        mp_face_det = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5,
        )

    # Load L2CS model
    print("  Loading L2CS-Net (batched mode)...")
    gaze_model, device = load_gaze_model(corpus_dir=args.corpus_dir)

    # Single pass: read frames, compute blink EAR, crop + run L2CS gaze in batches
    t_read = time.time()
    blink_lefts = np.zeros(total_frames, dtype=np.float32)
    blink_rights = np.zeros(total_frames, dtype=np.float32)
    gaze_pitches = np.zeros(total_frames, dtype=np.float32)
    gaze_yaws = np.zeros(total_frames, dtype=np.float32)
    valid_mask = np.zeros(total_frames, dtype=bool)

    batch_buffer = []
    batch_frame_ids = []
    n_crops = 0
    n_roi = 0         # frames where the cheap EMICA-ROI FaceMesh pass succeeded
    n_fullframe = 0   # frames that fell back to full-frame FaceMesh
    bs = args.batch_size

    for frame_idx in tqdm(range(total_frames), desc="Read + Blink + Gaze"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]

        # Fast path: run FaceMesh on the EMICA ROI. Landmarks come back in
        # full-frame pixels, so EAR is identical to the full-frame result.
        pts_px = None
        roi = (roi_from_bbox(emica_bbox_map[frame_idx], w, h)
               if frame_idx in emica_bbox_map else None)
        if roi is not None:
            rx0, ry0, rx1, ry1 = roi
            sub = np.ascontiguousarray(frame_rgb[ry0:ry1, rx0:rx1])
            roi_res = face_mesh.process(sub)
            if roi_res.multi_face_landmarks:
                pts_px = landmarks_to_px(roi_res.multi_face_landmarks[0],
                                         rx1 - rx0, ry1 - ry0, rx0, ry0)
                n_roi += 1

        if pts_px is None:  # fallback: full frame — same behaviour as before
            mp_results = face_mesh.process(frame_rgb)
            if mp_results.multi_face_landmarks:
                pts_px = landmarks_to_px(mp_results.multi_face_landmarks[0], w, h)
            n_fullframe += 1

        if pts_px is not None:
            blink_lefts[frame_idx] = compute_ear(pts_px, LEFT_EYE_EAR)
            blink_rights[frame_idx] = compute_ear(pts_px, RIGHT_EYE_EAR)
            valid_mask[frame_idx] = True

        bbox = None
        if frame_idx in emica_bbox_map:
            bbox = emica_bbox_map[frame_idx]
        elif pts_px is not None:
            margin = 0.15
            (x0, y0), (x1, y1) = pts_px.min(0), pts_px.max(0)
            bw, bh = x1 - x0, y1 - y0
            bbox = [x0 - bw * margin, y0 - bh * margin,
                    x1 + bw * margin, y1 + bh * margin]
        elif mp_face_det is not None:
            det_results = mp_face_det.process(frame_rgb)
            if det_results.detections:
                d = det_results.detections[0].location_data.relative_bounding_box
                bbox = [d.xmin * w, d.ymin * h,
                        (d.xmin + d.width) * w, (d.ymin + d.height) * h]

        if bbox is not None:
            # crop_face returns (3, H, W) float32 [0,1] — normalize directly
            cropped = crop_face(frame_rgb, bbox, GAZE_INPUT_SIZE)
            t = (torch.from_numpy(cropped) - _IMAGENET_MEAN) / _IMAGENET_STD
            batch_buffer.append(t)
            batch_frame_ids.append(frame_idx)
            valid_mask[frame_idx] = True
            n_crops += 1

            if len(batch_buffer) >= bs:
                _run_gaze_batch(batch_buffer, batch_frame_ids, gaze_model, device,
                                gaze_pitches, gaze_yaws)
                batch_buffer.clear()
                batch_frame_ids.clear()

    # Flush remaining batch
    if batch_buffer:
        _run_gaze_batch(batch_buffer, batch_frame_ids, gaze_model, device,
                        gaze_pitches, gaze_yaws)
        batch_buffer.clear()
        batch_frame_ids.clear()

    cap.release()
    face_mesh.close()
    if mp_face_det is not None:
        mp_face_det.close()

    print(f"  Read + blink + gaze: {time.time() - t_read:.1f}s ({n_crops} face crops)")
    print(f"  FaceMesh: {n_roi} on EMICA ROI (fast), {n_fullframe} full-frame (fallback)")

    # Build output
    valid_indices = np.where(valid_mask)[0]

    save_dict = {
        "gaze_pitch": gaze_pitches[valid_indices],
        "gaze_yaw": gaze_yaws[valid_indices],
        "blink_left": blink_lefts[valid_indices],
        "blink_right": blink_rights[valid_indices],
        "timestep_id": valid_indices.astype(np.int64),
    }

    np.savez_compressed(str(final_npz), **save_dict)
    dt = time.time() - t_start
    print(f"\n  [OK] Saved: {final_npz} ({final_npz.stat().st_size / 1024:.1f} KB)")
    print(f"  Frames processed: {len(valid_indices)}/{total_frames}")
    if len(valid_indices) > 0:
        vp, vy = save_dict["gaze_pitch"], save_dict["gaze_yaw"]
        bl, br = save_dict["blink_left"], save_dict["blink_right"]
        print(f"  Gaze range: pitch [{vp.min():.2f}, {vp.max():.2f}], "
              f"yaw [{vy.min():.2f}, {vy.max():.2f}]")
        print(f"  Blink EAR range: left [{bl.min():.3f}, {bl.max():.3f}], "
              f"right [{br.min():.3f}, {br.max():.3f}]")
    print(f"  Done ({dt:.1f}s)")


if __name__ == "__main__":
    main()
