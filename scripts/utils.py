"""Shared utilities for the vid2smplx pipeline.

Single source of truth for: rotation conversions, temporal smoothing,
video helpers, MANO/face utilities, renderer selection, and shared constants.
"""
from __future__ import annotations

import json
import pickle
import subprocess
from pathlib import Path

import io

import cv2
import numpy as np
import torch


def torch_load_buffered(path: str | Path, **kwargs) -> dict:
    """Load a PyTorch checkpoint via buffered read to avoid Lustre small-read latency.

    Reads the entire file into RAM in one sequential read (fast on Lustre),
    then deserializes from memory. 12x faster than torch.load under IO contention.
    """
    kwargs.setdefault("map_location", "cpu")
    kwargs.setdefault("weights_only", False)
    with open(path, "rb") as f:
        buf = io.BytesIO(f.read())
    return torch.load(buf, **kwargs)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HAMER_FOCAL_LENGTH = 5000.0
HAMER_IMAGE_SIZE = 256.0

MIN_KP_COUNT = 7
MIN_KP_CONF = 0.6

# MediaPipe eye landmark indices for EAR blink detection (6 points per eye)
LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]

# FLAME eyeball vertex indices (from Metrical Tracker)
LEFT_IRIS_FLAME = [4597, 4542, 4510, 4603, 4570]
RIGHT_IRIS_FLAME = [4051, 3996, 3964, 3932, 4028]

# MediaPipe iris landmark indices
LEFT_IRIS_MP = [468, 469, 470, 471, 472]
RIGHT_IRIS_MP = [473, 474, 475, 476, 477]

# Rendering defaults
CRF = 23
RENDER_SCALE = 0.5
MAX_RESOLUTION = 1920


# ---------------------------------------------------------------------------
# Rotation conversions (numpy)
# ---------------------------------------------------------------------------

def rotmat_to_axis_angle(rotmat: np.ndarray) -> np.ndarray:
    """Convert rotation matrix (3,3) to axis-angle (3,) via Rodrigues' formula."""
    rotmat = np.asarray(rotmat, dtype=np.float64)
    theta = np.arccos(np.clip((np.trace(rotmat) - 1) / 2, -1, 1))
    if theta < 1e-6:
        return np.zeros(3, dtype=np.float32)
    axis = np.array([
        rotmat[2, 1] - rotmat[1, 2],
        rotmat[0, 2] - rotmat[2, 0],
        rotmat[1, 0] - rotmat[0, 1],
    ])
    axis = axis / (2 * np.sin(theta))
    return (axis * theta).astype(np.float32)


def rotmat_batch_to_axis_angle(rotmats: np.ndarray) -> np.ndarray:
    """Convert (N, 3, 3) rotation matrices to (N, 3) axis-angle."""
    return np.stack([rotmat_to_axis_angle(r) for r in rotmats])


# ---------------------------------------------------------------------------
# Rotation conversions (torch)
# ---------------------------------------------------------------------------

def rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to 3x3 rotation matrix (torch)."""
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to axis-angle via quaternion intermediate (torch, batched)."""
    batch_shape = matrix.shape[:-2]
    m = matrix.reshape(-1, 3, 3)
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    quat = torch.zeros(m.shape[0], 4, device=m.device, dtype=m.dtype)

    s = torch.sqrt(torch.clamp(trace + 1, min=1e-10)) * 2
    mask = trace > 0
    quat[mask, 0] = 0.25 * s[mask]
    quat[mask, 1] = (m[mask, 2, 1] - m[mask, 1, 2]) / s[mask]
    quat[mask, 2] = (m[mask, 0, 2] - m[mask, 2, 0]) / s[mask]
    quat[mask, 3] = (m[mask, 1, 0] - m[mask, 0, 1]) / s[mask]

    cond1 = (~mask) & (m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2])
    s1 = torch.sqrt(torch.clamp(1 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2], min=1e-10)) * 2
    quat[cond1, 0] = (m[cond1, 2, 1] - m[cond1, 1, 2]) / s1[cond1]
    quat[cond1, 1] = 0.25 * s1[cond1]
    quat[cond1, 2] = (m[cond1, 0, 1] + m[cond1, 1, 0]) / s1[cond1]
    quat[cond1, 3] = (m[cond1, 0, 2] + m[cond1, 2, 0]) / s1[cond1]

    cond2 = (~mask) & (~cond1) & (m[:, 1, 1] > m[:, 2, 2])
    s2 = torch.sqrt(torch.clamp(1 + m[:, 1, 1] - m[:, 0, 0] - m[:, 2, 2], min=1e-10)) * 2
    quat[cond2, 0] = (m[cond2, 0, 2] - m[cond2, 2, 0]) / s2[cond2]
    quat[cond2, 1] = (m[cond2, 0, 1] + m[cond2, 1, 0]) / s2[cond2]
    quat[cond2, 2] = 0.25 * s2[cond2]
    quat[cond2, 3] = (m[cond2, 1, 2] + m[cond2, 2, 1]) / s2[cond2]

    cond3 = (~mask) & (~cond1) & (~cond2)
    s3 = torch.sqrt(torch.clamp(1 + m[:, 2, 2] - m[:, 0, 0] - m[:, 1, 1], min=1e-10)) * 2
    quat[cond3, 0] = (m[cond3, 1, 0] - m[cond3, 0, 1]) / s3[cond3]
    quat[cond3, 1] = (m[cond3, 0, 2] + m[cond3, 2, 0]) / s3[cond3]
    quat[cond3, 2] = (m[cond3, 1, 2] + m[cond3, 2, 1]) / s3[cond3]
    quat[cond3, 3] = 0.25 * s3[cond3]

    quat = quat * torch.sign(quat[:, :1])
    w = torch.clamp(quat[:, 0], -1.0, 1.0)
    angle = 2.0 * torch.acos(w)
    axis = quat[:, 1:]
    norm = torch.norm(axis, dim=-1, keepdim=True)
    small = (norm < 1e-8).squeeze(-1)
    axis = torch.where(small.unsqueeze(-1), torch.zeros_like(axis), axis / norm)
    return (axis * angle.unsqueeze(-1)).reshape(*batch_shape, 3)


def rot6d_to_axis_angle(rot6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation to axis-angle (torch)."""
    return matrix_to_axis_angle(rot6d_to_matrix(rot6d))


# ---------------------------------------------------------------------------
# Temporal smoothing
# ---------------------------------------------------------------------------

def one_euro_filter_np(x: np.ndarray, min_cutoff: float = 1.0, beta: float = 0.5, d_cutoff: float = 1.0, fps: float = 30.0) -> np.ndarray:
    """One-Euro temporal filter for numpy arrays. Shape: (L, ...) -> (L, ...)."""
    if x.shape[0] <= 1:
        return x
    dt = 1.0 / fps
    x_flat = x.reshape(x.shape[0], -1).copy()
    L, D = x_flat.shape

    def sf(cutoff):
        return 1.0 / (1.0 + 1.0 / (2 * np.pi * cutoff) / dt)

    out = np.zeros_like(x_flat)
    out[0] = x_flat[0]
    dx_prev = np.zeros(D)
    alpha_d = sf(d_cutoff)

    for i in range(1, L):
        dx = (x_flat[i] - x_flat[i - 1]) / dt
        dx_hat = alpha_d * dx + (1 - alpha_d) * dx_prev
        dx_prev = dx_hat
        alpha = sf(min_cutoff + beta * np.abs(dx_hat))
        out[i] = alpha * x_flat[i] + (1 - alpha) * out[i - 1]

    return out.reshape(x.shape)


def smooth_face_params(expr: np.ndarray, jaw: np.ndarray, eyes: np.ndarray, fps: float = 30.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply One-Euro filter with per-param tuning for face parameters."""
    expr_smooth = one_euro_filter_np(expr, min_cutoff=1.5, beta=0.3, fps=fps)
    jaw_smooth = one_euro_filter_np(jaw, min_cutoff=1.0, beta=0.5, fps=fps)
    eyes_smooth = one_euro_filter_np(eyes, min_cutoff=2.0, beta=0.2, fps=fps)
    return expr_smooth, jaw_smooth, eyes_smooth


# ---------------------------------------------------------------------------
# Video utilities
# ---------------------------------------------------------------------------

def get_video_fps(video_path: Path | str) -> float:
    """Get video FPS via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "json", str(video_path)],
        capture_output=True, text=True,
    )
    info = json.loads(result.stdout)
    fps_str = info["streams"][0]["r_frame_rate"]
    num, den = fps_str.split("/")
    return float(num) / float(den)


def ensure_max_resolution(video_path: Path | str, max_res: int = MAX_RESOLUTION) -> str:
    """Downscale video to max_res if needed. Returns (possibly new) path.

    Works both standalone (via ffprobe) and with GVHMR (via get_video_lwh).
    """
    video_path = str(video_path)
    try:
        from hmr4d.utils.video_io_utils import get_video_lwh
        _, width, height = get_video_lwh(video_path)
    except ImportError:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "json", video_path],
            capture_output=True, text=True,
        )
        info = json.loads(result.stdout)
        width = info["streams"][0]["width"]
        height = info["streams"][0]["height"]

    longer = max(width, height)
    if longer <= max_res:
        return video_path

    out_path = str(Path(video_path).with_suffix("")) + f"_{max_res}p.mp4"
    if Path(out_path).exists():
        print(f"  [Downscale] Using cached: {out_path}")
        return out_path
    print(f"  [Downscale] {width}x{height} -> {max_res}p for rendering...")
    scale_filter = f"scale='if(gt(iw,ih),{max_res},-2)':'if(gt(iw,ih),-2,{max_res})'"
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vf", scale_filter,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p",
        out_path, "-loglevel", "warning",
    ], check=True)
    print(f"  [Downscale] Done: {out_path}")
    return out_path


def crop_face(frame_rgb: np.ndarray, bbox: list[float], target_size: int = 224) -> np.ndarray:
    """Crop a square face region from a frame and resize.

    Args:
        frame_rgb: (H, W, 3) uint8 or float RGB image
        bbox: [x1, y1, x2, y2]
        target_size: output size

    Returns:
        cropped: (3, target_size, target_size) float32 array in [0, 1]
    """
    h, w = frame_rgb.shape[:2]
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    size = max(x2 - x1, y2 - y1)
    half = size / 2

    sx1, sy1 = int(cx - half), int(cy - half)
    sx2, sy2 = sx1 + int(size), sy1 + int(size)

    px1 = max(0, -sx1)
    py1 = max(0, -sy1)
    px2 = max(0, sx2 - w)
    py2 = max(0, sy2 - h)

    crop = frame_rgb[max(0, sy1):min(h, sy2), max(0, sx1):min(w, sx2)]

    if px1 > 0 or py1 > 0 or px2 > 0 or py2 > 0:
        crop = cv2.copyMakeBorder(crop, py1, py2, px1, px2, cv2.BORDER_CONSTANT, value=0)

    resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_CUBIC)

    if resized.dtype == np.uint8:
        resized = resized / 255.0

    return resized.transpose(2, 0, 1).astype(np.float32)


def expand_bbox(bbox: list[float], scale: float = 1.3) -> list[float]:
    """Expand a bbox by a scale factor around its center."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1
    nw, nh = w * scale, h * scale
    return [cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2]


# ---------------------------------------------------------------------------
# EMICA batch size auto-tuning
# ---------------------------------------------------------------------------

def auto_emica_batch_size(model: torch.nn.Module, sample_image: torch.Tensor, target_util: float = 0.85) -> int:
    """Probe GPU memory to find optimal EMICA batch size."""
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # Probe with bs=1 to get fixed overhead
    batch = {"image": sample_image.unsqueeze(0).cuda()}
    with torch.no_grad():
        _ = model(batch, training=False, validation=False)
    peak1 = torch.cuda.max_memory_allocated(device)
    del batch, _
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # Probe with bs=4 to get marginal cost
    test_bs = 4
    batch = {"image": sample_image.unsqueeze(0).expand(test_bs, -1, -1, -1).cuda()}
    with torch.no_grad():
        _ = model(batch, training=False, validation=False)
    peak4 = torch.cuda.max_memory_allocated(device)
    per_sample = (peak4 - peak1) / (test_bs - 1)
    del batch, _
    torch.cuda.empty_cache()

    total = torch.cuda.get_device_properties(device).total_memory
    available = total * target_util - peak1
    optimal = max(1, min(512, int(available / max(per_sample, 1))))
    if total < 12e9:
        optimal = max(1, optimal // 2)
    print(f"  [Auto BS] EMICA: {total/1e9:.1f}GB GPU, {per_sample/1e6:.0f}MB/sample -> bs={optimal}")
    return optimal


# ---------------------------------------------------------------------------
# Renderer selection
# ---------------------------------------------------------------------------

def get_renderer_class() -> type:
    """Return the best available Renderer class.

    Prefers nvdiffrast (faster) when RENDERER=nvdr and hardware supports it,
    falls back to pytorch3d Renderer instantly if not compatible.
    """
    import os
    from hmr4d.utils.vis.renderer import Renderer as PT3DRenderer

    if os.environ.get("RENDERER") != "nvdr":
        return PT3DRenderer

    try:
        import nvdiffrast.torch as dr
        ctx = dr.RasterizeCudaContext()
        del ctx
        from hmr4d.utils.vis.renderer_nvdr import NvDiffRastRenderer
        print("[RENDERER] Using nvdiffrast (NvDiffRastRenderer)")
        return NvDiffRastRenderer
    except Exception as e:
        print(f"[RENDERER] nvdiffrast unavailable ({e}), falling back to pytorch3d")
        return PT3DRenderer


# ---------------------------------------------------------------------------
# MANO utilities
# ---------------------------------------------------------------------------

def _get_repo_dir():
    """Return the repo root directory."""
    return Path(__file__).resolve().parent.parent


def load_mano_faces() -> np.ndarray | None:
    """Load MANO faces from MANO_RIGHT.pkl. Returns (F,3) int64 array."""
    mano_pkl = _get_repo_dir() / "models" / "mano" / "MANO_RIGHT.pkl"
    if not mano_pkl.exists():
        print(f"ERROR: MANO model not found at {mano_pkl}")
        print("Download from https://mano.is.tue.mpg.de/ and extract to models/mano/")
        return None
    with open(mano_pkl, "rb") as f:
        mano_data = pickle.load(f, encoding="latin1")
    return np.array(mano_data["f"], dtype=np.int64)


def load_mano_mean_hand_pose() -> np.ndarray:
    """Load MANO mean hand pose (relaxed). Returns (45,) axis-angle."""
    repo_dir = _get_repo_dir()
    candidates = [
        repo_dir / "models" / "mano" / "MANO_RIGHT.pkl",
        repo_dir / "mano_v1_2" / "models" / "MANO_RIGHT.pkl",
    ]
    mano_pkl = next((p for p in candidates if p.exists()), None)
    if mano_pkl is None:
        return np.zeros(45, dtype=np.float32)
    with open(mano_pkl, "rb") as f:
        mano_data = pickle.load(f, encoding="latin1")
    hands_mean = mano_data.get("hands_mean", None)
    if hands_mean is None:
        return np.zeros(45, dtype=np.float32)
    return np.array(hands_mean, dtype=np.float32).flatten()[:45]


def load_mano_joint_regressor() -> torch.Tensor:
    """Load MANO joint regressor from MANO_RIGHT.pkl. Returns (16, 778) float tensor."""
    mano_pkl = _get_repo_dir() / "models" / "mano" / "MANO_RIGHT.pkl"
    with open(mano_pkl, "rb") as f:
        mano_data = pickle.load(f, encoding="latin1")
    J = mano_data["J_regressor"]
    if hasattr(J, "toarray"):
        J = J.toarray()
    return torch.from_numpy(np.array(J, dtype=np.float32))


# ---------------------------------------------------------------------------
# SMPL-X chunked forward pass
# ---------------------------------------------------------------------------

def smplx_forward_chunked(model: torch.nn.Module, params: dict[str, torch.Tensor], chunk_size: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    """Run SMPL-X forward pass in chunks to avoid OOM.

    Args:
        model: SMPL-X model instance (on GPU).
        params: dict of param tensors, each (N, D). Missing optional params
                are padded with zeros.
        chunk_size: max frames per forward pass.

    Returns:
        (verts, joints): concatenated tensors on CPU, shapes (N, V, 3) and (N, J, 3).
    """
    n = next(iter(params.values())).shape[0]

    # Pad missing optional params with zeros
    defaults = {
        "left_hand_pose": 45, "right_hand_pose": 45,
        "jaw_pose": 3, "expression": 100, "leye_pose": 3, "reye_pose": 3,
    }
    for key, dim in defaults.items():
        if key not in params:
            params[key] = torch.zeros(n, dim)

    all_verts, all_joints = [], []
    for i in range(0, n, chunk_size):
        chunk = {k: v[i:i + chunk_size].cuda() for k, v in params.items()}
        model.batch_size = next(iter(chunk.values())).shape[0]
        with torch.no_grad():
            out = model(**chunk)
        all_verts.append(out.vertices.cpu())
        all_joints.append(out.joints.cpu())

    return torch.cat(all_verts, 0), torch.cat(all_joints, 0)
