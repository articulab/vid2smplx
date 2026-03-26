#!/usr/bin/env python3
"""EMICA face reconstruction using MediaPipe face detection (streaming).

Single-pass face detection + crop, batched EMICA inference, streaming eye pose
solve. No disk frame extraction — reads video directly via cv2.

Output: <out_folder>/<video_name>/flame_params.npz

Usage:
    python scripts/run_emica.py --video input.mp4 --out_folder output/emica
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from utils import (
    get_video_fps, crop_face, expand_bbox, rot6d_to_axis_angle,
    auto_emica_batch_size, smooth_face_params,
    LEFT_IRIS_FLAME, RIGHT_IRIS_FLAME, LEFT_IRIS_MP, RIGHT_IRIS_MP,
)


# ---------------------------------------------------------------------------
# Face detection + crop (single video pass, streaming)
# ---------------------------------------------------------------------------

def detect_and_crop_streaming(video_path: Path | str, crop_size: int = 224, scale: float = 1.3) -> tuple[list[str], int, np.ndarray, list[int]]:
    """Single video pass: MediaPipe face detect + crop, one frame at a time.

    Returns:
        images: (N, 3, crop_size, crop_size) float32 tensor
        bboxes: (N, 4) float32 array [x1, y1, x2, y2] (expanded)
        valid_indices: list of frame indices with detections
    """
    import mediapipe as mp

    face_detection = mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bboxes = []
    valid_indices = []
    last_bbox = None

    # Accumulate crops in a fixed-size buffer, flush to disk periodically.
    # This keeps RAM usage at ~buffer_size * 3 * 224 * 224 * 4 bytes.
    import tempfile, os
    BUFFER_SIZE = 1024  # ~230MB per buffer
    buffer = []
    tmp = tempfile.NamedTemporaryFile(suffix=".dat", delete=False)
    tmp_path = tmp.name
    tmp.close()
    os.unlink(tmp_path)  # just need the unique prefix
    chunk_paths = []
    n_detected = 0

    for idx in tqdm(range(total_frames), desc="Detect + crop faces"):
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        results = face_detection.process(img_rgb)

        bbox = None
        if results.detections:
            det = results.detections[0]
            bb = det.location_data.relative_bounding_box
            bbox = [bb.xmin * w, bb.ymin * h,
                    (bb.xmin + bb.width) * w, (bb.ymin + bb.height) * h]
            bbox = expand_bbox(bbox, scale)
            last_bbox = bbox
        elif last_bbox is not None:
            bbox = last_bbox

        if bbox is None:
            continue

        cropped = crop_face(img_rgb, bbox, crop_size)
        buffer.append(cropped)
        bboxes.append(bbox)
        valid_indices.append(idx)
        n_detected += 1

        if len(buffer) >= BUFFER_SIZE:
            chunk_path = f"{tmp_path}.chunk{len(chunk_paths)}.npy"
            np.save(chunk_path, np.stack(buffer))
            chunk_paths.append(chunk_path)
            buffer.clear()

    # Flush remaining
    if buffer:
        chunk_path = f"{tmp_path}.chunk{len(chunk_paths)}.npy"
        np.save(chunk_path, np.stack(buffer))
        chunk_paths.append(chunk_path)
        buffer.clear()

    cap.release()
    face_detection.close()

    bboxes_np = np.array(bboxes, dtype=np.float32) if bboxes else np.zeros((0, 4), dtype=np.float32)
    return chunk_paths, n_detected, bboxes_np, valid_indices


# ---------------------------------------------------------------------------
# EMICA inference
# ---------------------------------------------------------------------------

def run_emica_inference(chunk_paths: list[str], n_total: int, iris_indices: list[int], batch_size: int = 32) -> tuple[dict[str, torch.Tensor | np.ndarray], list[str], list[str]]:
    """Run EMICA model on face crops stored as .npy chunk files.

    Loads one chunk at a time to avoid holding all crops in RAM.
    Full verts are saved to temporary .npy chunk files on disk (for later NPZ
    save) while only iris vertices are kept in RAM.

    Args:
        chunk_paths: list of paths to .npy files, each (chunk_size, 3, 224, 224)
        n_total: total number of face crops across all chunks
        iris_indices: list of FLAME vertex indices for iris (left + right)
        batch_size: GPU batch size

    Returns:
        (all_results, verts_chunk_paths, input_chunk_paths)
        all_results: dict with expcode, jawpose, globalpose, cam, shapecode,
                     iris_verts, faces
        verts_chunk_paths: list of .npy paths holding full verts chunks
        input_chunk_paths: the input chunk_paths (caller deletes after NPZ save)
    """
    import os, tempfile
    from inferno.utils.other import get_path_to_assets
    from inferno_apps.FaceReconstruction.utils.load import load_model

    path_to_models = str(Path(get_path_to_assets()) / "FaceReconstruction/models")
    model_name = "EMICA-CVT_flame2020_notexture"

    print(f"  Loading EMICA model: {model_name}")
    face_rec_model, conf = load_model(path_to_models, model_name, with_losses=False)
    face_rec_model.cuda()
    face_rec_model.eval()

    # Auto batch size from first chunk
    if batch_size <= 32 and n_total > 0 and torch.cuda.is_available():
        first_chunk = torch.from_numpy(np.load(chunk_paths[0]))
        batch_size = auto_emica_batch_size(face_rec_model, first_chunk[0])
        del first_chunk

    all_results = {
        "expcode": [], "jawpose": [], "globalpose": [],
        "cam": [], "shapecode": [], "iris_verts": [],
    }
    flame_faces = None
    verts_chunk_paths = []
    verts_buffer = []

    print(f"  Running EMICA on {n_total} frames (batch_size={batch_size})...")
    n_done = 0
    with torch.no_grad():
        for chunk_path in chunk_paths:
            # Load one chunk at a time (~230MB for 1024 crops)
            chunk_data = torch.from_numpy(np.load(chunk_path))
            # Don't delete chunk_path here — caller handles cleanup

            for start in range(0, len(chunk_data), batch_size):
                end = min(start + batch_size, len(chunk_data))
                batch = {"image": chunk_data[start:end].cuda()}
                values = face_rec_model(batch, training=False, validation=False)

                all_results["expcode"].append(values["expcode"].cpu())
                all_results["jawpose"].append(values["jawpose"].cpu())
                gp = values["globalpose"].cpu()
                if gp.shape[-1] == 6:
                    gp = rot6d_to_axis_angle(gp)
                all_results["globalpose"].append(gp)
                all_results["cam"].append(values["cam"].cpu())
                all_results["shapecode"].append(values["shapecode"].cpu())

                full_verts = values["verts"].cpu()
                verts_buffer.append(full_verts.numpy())
                all_results["iris_verts"].append(full_verts[:, iris_indices])

                if flame_faces is None:
                    for attr_path in [
                        ("flame", "faces_tensor"),
                        ("shape_model", "faces_tensor"),
                        ("shape_model", "flame", "faces_tensor"),
                    ]:
                        obj = face_rec_model
                        try:
                            for a in attr_path:
                                obj = getattr(obj, a)
                            flame_faces = obj.cpu().numpy().astype(np.int64)
                            break
                        except AttributeError:
                            continue

                n_done += end - start
                print(f"\r  EMICA: {n_done}/{n_total}", end="", flush=True)

            del chunk_data

            # Flush verts buffer to disk after each input chunk
            if verts_buffer:
                fd, verts_path = tempfile.mkstemp(suffix=".npy", prefix="emica_verts_")
                os.close(fd)
                np.save(verts_path, np.concatenate(verts_buffer, axis=0))
                verts_chunk_paths.append(verts_path)
                verts_buffer.clear()

    print()  # newline after progress

    for k in all_results:
        all_results[k] = torch.cat(all_results[k], dim=0)

    if flame_faces is None:
        flame_path = Path(get_path_to_assets()) / "FaceReconstruction/flame/generic_model.pkl"
        if flame_path.exists():
            import pickle
            with open(flame_path, "rb") as f:
                flame_faces = np.array(pickle.load(f, encoding="latin1")["f"], dtype=np.int64)
        else:
            flame_faces = np.zeros((0, 3), dtype=np.int64)

    all_results["faces"] = flame_faces
    return all_results, verts_chunk_paths, list(chunk_paths)


# ---------------------------------------------------------------------------
# Eye pose solve (streaming, 2nd video pass)
# ---------------------------------------------------------------------------

def solve_eye_poses_streaming(video_path: Path | str, iris_verts: torch.Tensor, emica_cam: torch.Tensor, face_bboxes: np.ndarray,
                              valid_indices: list[int]) -> np.ndarray:
    """Solve eye rotations reading frames one-at-a-time from video.

    Args:
        iris_verts: (N, 10, 3) tensor — first 5 = left iris, last 5 = right iris
    """
    import mediapipe as mp

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5,
    )

    cap = cv2.VideoCapture(str(video_path))
    N = len(valid_indices)
    eyes_pose = np.zeros((N, 6), dtype=np.float32)

    valid_set = {idx: i for i, idx in enumerate(valid_indices)}

    for frame_idx in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), desc="Eye poses"):
        if frame_idx not in valid_set:
            cap.grab()
            continue

        ret, frame = cap.read()
        if not ret:
            break

        i = valid_set[frame_idx]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        results = face_mesh.process(img_rgb)
        if not results.multi_face_landmarks:
            continue

        landmarks = results.multi_face_landmarks[0]
        bbox = face_bboxes[i]
        verts = iris_verts[i].numpy()  # (10, 3) — left 0:5, right 5:10
        cam = emica_cam[i].numpy()

        for side, iris_mp, iris_local, offset in [
            ("left", LEFT_IRIS_MP, slice(0, 5), 0),
            ("right", RIGHT_IRIS_MP, slice(5, 10), 3),
        ]:
            target_2d = np.array([
                [landmarks.landmark[li].x * w, landmarks.landmark[li].y * h]
                for li in iris_mp
            ], dtype=np.float32)

            eye_verts = verts[iris_local]
            s, tx, ty = cam
            x_norm = (s * eye_verts[:, 0] + tx + 1) / 2
            y_norm = (s * eye_verts[:, 1] + ty + 1) / 2
            x1, y1, x2, y2 = bbox
            proj_2d = np.stack([x1 + x_norm * (x2 - x1), y1 + y_norm * (y2 - y1)], axis=-1)

            delta = target_2d.mean(axis=0) - proj_2d.mean(axis=0)
            bbox_size = max(x2 - x1, y2 - y1)
            if bbox_size > 0:
                eyes_pose[i, offset:offset + 3] = [
                    -delta[1] / bbox_size * 2.0,
                    delta[0] / bbox_size * 2.0,
                    0.0,
                ]

    cap.release()
    face_mesh.close()
    return eyes_pose


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="EMICA face reconstruction (streaming)")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--out_folder", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--bbox_scale", type=float, default=1.3)
    args = parser.parse_args()

    t_start = time.time()

    video_path = Path(args.video).resolve()
    video_name = video_path.stem
    out_dir = Path(args.out_folder).resolve() / video_name
    out_dir.mkdir(parents=True, exist_ok=True)

    final_npz = out_dir / "flame_params.npz"
    if final_npz.exists():
        print(f"  [SKIP] Already exists: {final_npz}")
        return

    print(f"=== EMICA Face Reconstruction ===")
    print(f"Video:      {video_path}")
    print(f"Output:     {out_dir}")
    print()

    fps = get_video_fps(str(video_path))
    print(f"  Video FPS: {fps:.1f}")

    # Step 1: Detect faces + crop (streaming, chunked to disk)
    print(f"\n  [Step 1] MediaPipe face detection + crop (streaming)...")
    chunk_paths, n_detected, bboxes, valid_indices = detect_and_crop_streaming(
        video_path, scale=args.bbox_scale,
    )
    print(f"  Detected faces in {len(valid_indices)} frames ({len(chunk_paths)} chunks)")

    if n_detected == 0:
        print("  [ERROR] No faces detected in any frame!")
        sys.exit(1)

    # Save detection cache for gaze_blink reuse
    detection_cache = out_dir / "_detection_cache.npz"
    if not detection_cache.exists():
        np.savez_compressed(detection_cache,
                            bboxes=bboxes, valid_indices=np.array(valid_indices))

    # Step 2: EMICA inference (reads chunks from disk, never holds all crops in RAM)
    print(f"\n  [Step 2] EMICA inference...")
    iris_indices = LEFT_IRIS_FLAME + RIGHT_IRIS_FLAME
    emica_results, verts_chunk_paths, input_chunk_paths = run_emica_inference(
        chunk_paths, n_detected, iris_indices, batch_size=args.batch_size,
    )

    expr = emica_results["expcode"].numpy()
    jaw = emica_results["jawpose"].numpy()
    globalpose = emica_results["globalpose"].numpy()
    cam = emica_results["cam"].numpy()
    shape = emica_results["shapecode"].numpy()
    faces = emica_results["faces"]
    print(f"  EMICA output: expr {expr.shape}, jaw {jaw.shape}, iris_verts {emica_results['iris_verts'].shape}")

    # Step 3: Eye pose solve (streaming, uses iris_verts only)
    print(f"\n  [Step 3] MediaPipe eye pose solve (streaming)...")
    eyes_pose = solve_eye_poses_streaming(
        video_path, emica_results["iris_verts"], emica_results["cam"],
        bboxes, valid_indices,
    )
    print(f"  Eyes pose: {eyes_pose.shape}")

    # Step 4: Temporal smoothing
    print(f"\n  [Step 4] Temporal smoothing...")
    expr, jaw, eyes_pose = smooth_face_params(expr, jaw, eyes_pose, fps=fps)

    # Step 5: Load full verts from disk chunks, save NPZ, then clean up
    print(f"\n  [Step 5] Saving...")
    verts = np.concatenate([np.load(p) for p in verts_chunk_paths], axis=0)
    save_dict = {
        "expr": expr.astype(np.float32),
        "jaw_pose": jaw.astype(np.float32),
        "eyes_pose": eyes_pose.astype(np.float32),
        "globalpose": globalpose.astype(np.float32),
        "shape": shape.astype(np.float32),
        "cam": cam.astype(np.float32),
        "verts": verts.astype(np.float32),
        "face_bbox": bboxes.astype(np.float32),
        "timestep_id": np.array(valid_indices, dtype=np.int64),
        "faces": faces.astype(np.int64),
    }
    np.savez_compressed(str(final_npz), **save_dict)

    # Clean up all temp chunk files
    import os
    for p in input_chunk_paths + verts_chunk_paths:
        try:
            os.unlink(p)
        except OSError:
            pass
    dt = time.time() - t_start
    print(f"\n  [OK] Saved: {final_npz} ({final_npz.stat().st_size / 1024:.1f} KB)")
    print(f"  Total: {dt:.1f}s")


if __name__ == "__main__":
    main()
