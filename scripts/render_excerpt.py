#!/usr/bin/env python3
"""Render a short excerpt at the highest-motion window: original | reconstruction.

Motion score = per-frame joint-velocity magnitude over arms+hands (the parts that
actually carry gesture), smoothed, then the best contiguous window is chosen.

    python render_excerpt.py --npz clip/smplx_params.npz --video in.mp4 \
                             --out excerpt.mp4 [--seconds 60] [--fps 30]
"""
import argparse
import os
import subprocess
import time

import numpy as np

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import cv2  # noqa: E402
import pyrender  # noqa: E402
import smplx  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--npz", required=True)
ap.add_argument("--video", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--smplx_dir", default="/scratch/ymachta/Corpus/vid2smplx/models")
ap.add_argument("--seconds", type=float, default=60.0)
ap.add_argument("--fps", type=float, default=30.0)
ap.add_argument("--scale", type=float, default=0.5)
ap.add_argument("--expand", type=float, default=0.0,
                help="Widen the virtual camera by this fraction of frame size on each "
                     "side, so mesh outside the original frame (e.g. hands below the "
                     "bottom edge) is still rendered. 0 = match the video exactly.")
ap.add_argument("--no_crop", action="store_true",
                help="Skip the crop-to-subject step and show the whole (expanded) frame.")
args = ap.parse_args()

z = np.load(args.npz, allow_pickle=True)
L = int(z["num_frames"])
N = min(L, int(round(args.seconds * args.fps)))

# ---- pick the highest-motion window -----------------------------------------
# arms are body_pose joints 15..20; add hands so gesture (not just translation) counts
bp = z["body_pose"].reshape(L, 21, 3)
sig = [bp[:, 15:21].reshape(L, -1)]
for k in ("left_hand_pose", "right_hand_pose"):
    if k in z:
        sig.append(z[k].reshape(L, -1))
sig = np.concatenate(sig, axis=1)
vel = np.abs(np.diff(sig, axis=0)).sum(axis=1)
vel = np.concatenate([[0.0], vel])

valid = z["body_valid"].astype(bool) if "body_valid" in z else np.ones(L, bool)
vel = vel * valid  # never score a window on frames the pipeline flagged bad

k = np.ones(N) / N
score = np.convolve(vel, k, mode="valid")
start = int(np.argmax(score))
print(f"[excerpt] {L} frames; best {N}-frame window at {start} "
      f"({start/args.fps:.0f}s), motion={score[start]:.4f} vs mean={score.mean():.4f}",
      flush=True)

# ---- set up renderer ---------------------------------------------------------
K = z["K_fullimg"][0]
W, H = int(K[0, 2] * 2), int(K[1, 2] * 2)
S = args.scale
# Expanding the canvas while keeping fx/fy and shifting the principal point widens
# the frustum: same perspective, more of the world visible past the frame edge.
MX, MY = int(W * args.expand), int(H * args.expand)
EW, EH = W + 2 * MX, H + 2 * MY
rw, rh = int(EW * S), int(EH * S)

model = smplx.create(args.smplx_dir, model_type="smplx", gender="neutral",
                     use_face_contour=False, num_betas=10, num_expression_coeffs=100,
                     use_pca=False, flat_hand_mean=True, batch_size=1).cuda()
faces = model.faces.astype(np.int32)


def fk(i):
    t = lambda key: torch.from_numpy(z[key][i:i + 1]).float().cuda()
    with torch.no_grad():
        return model(body_pose=t("body_pose"), global_orient=t("global_orient_incam"),
                     betas=t("betas"), transl=t("transl_incam"),
                     left_hand_pose=t("left_hand_pose"), right_hand_pose=t("right_hand_pose"),
                     jaw_pose=t("jaw_pose"), leye_pose=t("leye_pose"),
                     reye_pose=t("reye_pose"), expression=t("expression")).vertices[0].cpu().numpy()


# crop to subject (p5/p95 so bad-transl frames don't blow the box out)
boxes = []
for i in range(start, start + N, max(1, N // 40)):
    v = fk(i)
    v = v[v[:, 2] > 1e-3]
    if len(v) < 100:
        continue
    uv = (K[:2, :2] @ (v[:, :2] / v[:, 2:3]).T).T + K[:2, 2] + [MX, MY]
    uv = uv[np.isfinite(uv).all(1)]
    if len(uv) >= 100:
        boxes.append(np.r_[uv.min(0), uv.max(0)])
b = np.array(boxes)
lo = np.array([np.percentile(b[:, 0], 5), np.percentile(b[:, 1], 5)])
hi = np.array([np.percentile(b[:, 2], 95), np.percentile(b[:, 3], 95)])
pad = 0.15 * max(hi - lo)
if args.no_crop:
    sx0, sy0, sx1, sy1 = 0, 0, rw, rh
else:
    sx0, sy0 = int(max(0, lo[0] - pad) * S), int(max(0, lo[1] - pad) * S)
    sx1, sy1 = int(min(EW, hi[0] + pad) * S), int(min(EH, hi[1] + pad) * S)
    sx1, sy1 = min(rw, max(sx0 + 16, sx1)), min(rh, max(sy0 + 16, sy1))
# libx264 + yuv420p requires EVEN width and height. An odd crop makes the encoder
# fail with "Error while opening encoder", which silently cost 19 excerpts before
# this was tracked down. Panels are concatenated horizontally, so each panel's width
# must be even too (2*even stays even, but 2*odd would also be even and hide an odd
# panel — so force both dimensions even here).
sx1 -= (sx1 - sx0) % 2
sy1 -= (sy1 - sy0) % 2
print(f"[setup] canvas {rw}x{rh} (expand={args.expand}), crop [{sx0}:{sx1}, {sy0}:{sy1}] "
      f"-> {(sx1 - sx0) * 2}x{sy1 - sy0} output (even-checked)", flush=True)

cam = pyrender.IntrinsicsCamera(fx=K[0, 0] * S, fy=K[1, 1] * S,
                                cx=(K[0, 2] + MX) * S, cy=(K[1, 2] + MY) * S,
                                znear=0.05, zfar=50.0)
cam_pose = np.diag([1.0, -1.0, -1.0, 1.0])   # OpenCV -> OpenGL
renderer = pyrender.OffscreenRenderer(rw, rh)

# NOTE: no cap.set() — CAP_PROP_POS_FRAMES is keyframe-inaccurate and desyncs
# video from mesh. Sequential grab is ground truth and costs ~10s.
cap = cv2.VideoCapture(args.video)
for _ in range(start):
    if not cap.grab():
        break

fdir = os.path.join(os.path.dirname(os.path.abspath(args.out)), "_ex")
os.makedirs(fdir, exist_ok=True)
t0 = time.time()
tick = max(1, N // 10)

for n in range(N):
    if n and n % tick == 0:
        el = time.time() - t0
        print(f"[progress] {n}/{N} ({100*n/N:.0f}%) {n/el:.1f} fps "
              f"eta {el/n*(N-n):.0f}s", flush=True)
    ok, og = cap.read()
    if ok:
        vw, vh = int(W * S), int(H * S)
        small = cv2.resize(og, (vw, vh))
        og = np.full((rh, rw, 3), 40, np.uint8)     # grey = outside the real frame
        oy, ox = int(MY * S), int(MX * S)
        og[oy:oy + vh, ox:ox + vw] = small
    else:
        og = np.zeros((rh, rw, 3), np.uint8)

    sc = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[0.4] * 3)
    sc.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(fk(start + n), faces, process=False),
                                      smooth=True))
    sc.add(cam, pose=cam_pose)
    sc.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=cam_pose)
    col, _ = renderer.render(sc)
    mesh_bgr = cv2.cvtColor(col[..., :3], cv2.COLOR_RGB2BGR)

    strip = np.concatenate([og[sy0:sy1, sx0:sx1], mesh_bgr[sy0:sy1, sx0:sx1]], axis=1)
    cv2.imwrite(f"{fdir}/{n:05d}.png", strip)

cap.release()
renderer.delete()
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{args.fps}",
                "-i", f"{fdir}/%05d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "20", args.out], check=True)
subprocess.run(["rm", "-rf", fdir])
print(f"[done] {args.out} in {time.time()-t0:.0f}s", flush=True)
