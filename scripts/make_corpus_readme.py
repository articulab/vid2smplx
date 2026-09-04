#!/usr/bin/env python3
"""Generate README.md for the interpersonality output corpus.

Regenerable: reads the actual .npz files and writes a README describing the tree,
the array shapes, and the pipeline that produced them. Re-run as more clips land.

    python make_corpus_readme.py [--root ~/staging/interpersonality_out]
"""
import argparse
import collections
import datetime
import glob
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.path.expanduser("~/staging/interpersonality_out"))
ap.add_argument("--fps", type=float, default=29.97)
args = ap.parse_args()
R = args.root

files = sorted(glob.glob(f"{R}/*/*/*.npz"))
if not files:
    raise SystemExit(f"no .npz under {R}")

clips, by_cond, by_sess = [], collections.defaultdict(list), collections.defaultdict(list)
keys_seen, shapes = None, {}
for f in files:
    z = np.load(f, allow_pickle=True)
    rel = os.path.relpath(f, R)
    sess, cond = rel.split(os.sep)[0], rel.split(os.sep)[1]
    name = os.path.basename(f)[:-4]
    L = int(z["num_frames"])
    exc = os.path.exists(f"{os.path.dirname(f)}/{name}_excerpt.mp4")
    fps = float(z["fps"]) if "fps" in z and float(z["fps"]) > 0 else args.fps
    rec = dict(name=name, sess=sess, cond=cond, frames=L, fps=fps,
               mb=os.path.getsize(f) / 1e6, excerpt=exc,
               hl=float(z["left_hand_valid"].mean()) if "left_hand_valid" in z else 0.0,
               hr=float(z["right_hand_valid"].mean()) if "right_hand_valid" in z else 0.0,
               face=float(z["face_valid"].mean()) if "face_valid" in z else 0.0)
    clips.append(rec)
    by_cond[cond].append(rec)
    by_sess[sess].append(rec)
    if keys_seen is None:
        keys_seen = sorted(z.keys())
        for k in keys_seen:
            a = z[k]
            shapes[k] = (str(a.shape) if a.ndim else "scalar", str(a.dtype))

tot_frames = sum(c["frames"] for c in clips)
tot_mb = sum(c["mb"] for c in clips)
n_exc = sum(c["excerpt"] for c in clips)

L = []
w = L.append
w("# Interpersonality SMPL-X corpus")
w("")
w(f"_Generated {datetime.date.today().isoformat()} by "
  "`vid2smplx/scripts/make_corpus_readme.py` — re-run to refresh._")
w("")
tot_hours = sum(c["frames"] / c["fps"] for c in clips) / 3600
w(f"**{len(clips)} clips**, {tot_frames:,} frames "
  f"({tot_hours:.1f} h of video), {tot_mb / 1000:.1f} GB. "
  f"{n_exc}/{len(clips)} have a preview excerpt.")
w("")
w("Each clip is one participant's camera. Participants in the same session and")
w("condition are recorded simultaneously, so they share a frame count and can be")
w("paired for dyadic analysis.")
w("")

w("## Layout")
w("")
w("Mirrors the source dataset (`S{NN}/{CONDITION}/`):")
w("")
w("```")
w(f"{os.path.basename(R)}/")
for s in sorted(by_sess):
    w(f"├── {s}/")
    conds = sorted({c['cond'] for c in by_sess[s]})
    for ci, cond in enumerate(conds):
        recs = sorted([c for c in by_sess[s] if c["cond"] == cond], key=lambda r: r["name"])
        branch = "└──" if ci == len(conds) - 1 else "├──"
        w(f"│   {branch} {cond}/    {len(recs)} clips, {recs[0]['frames']:,} frames each"
          if len({r['frames'] for r in recs}) == 1 else
          f"│   {branch} {cond}/    {len(recs)} clips")
        if s == sorted(by_sess)[0] and ci == 0:
            r = recs[0]
            w(f"│   │        ├── {r['name']}.npz            SMPL-X params, ALL {r['frames']:,} frames")
            w(f"│   │        ├── {r['name']}_excerpt.mp4    60s preview (original | reconstruction)")
            w(f"│   │        └── {r['name']}_summary.json   coverage + per-stage timings")
w("```")
w("")

w("## Clip lengths")
w("")
rates = sorted({round(c["fps"], 2) for c in clips})
w(f"Frame counts are **not uniform**, and the corpus mixes frame rates ({rates}). "
  "Each clip stores its own `fps`; **do not assume a single rate** — frame index is "
  "only a time base per clip.")
w("")
w("| condition | clips | frames (min–max) | duration | npz size |")
w("|---|---:|---|---|---|")
for cond in sorted(by_cond):
    rs = by_cond[cond]
    fs = [r["frames"] for r in rs]
    rng = f"{min(fs):,}" if min(fs) == max(fs) else f"{min(fs):,} – {max(fs):,}"
    w(f"| {cond} | {len(rs)} | {rng} | "
      f"{sum(r['frames'] / r['fps'] for r in rs) / len(rs) / 60:.1f} min avg | {sum(r['mb'] for r in rs) / len(rs):.1f} MB |")
w("")
w("BP / CS / TP are the long (~20 min) conditions; FT1 / FT2 are short (~7 min).")
w("")

w("## What's in a .npz")
w("")
w(f"Every array is full-length — one row per frame of the entire video (not just the")
w(f"excerpt window). Shapes below are from `{clips[0]['name']}` (L = {clips[0]['frames']:,} frames).")
w("")
w("| key | shape | dtype | meaning |")
w("|---|---|---|---|")
MEAN = {
    "body_pose": "21 SMPL-X body joints, axis-angle (L, 21x3)",
    "global_orient": "root orientation in world frame",
    "global_orient_incam": "root orientation in camera frame",
    "transl": "root translation, world frame",
    "transl_incam": "root translation, camera frame",
    "betas": "body shape coefficients (constant per clip)",
    "left_hand_pose": "15 MANO finger joints, axis-angle",
    "right_hand_pose": "15 MANO finger joints, axis-angle",
    "left_hand_valid": "per-frame: was the left hand actually observed",
    "right_hand_valid": "per-frame: was the right hand actually observed",
    "jaw_pose": "FLAME jaw rotation",
    "leye_pose": "left eyeball rotation",
    "reye_pose": "right eyeball rotation",
    "expression": "FLAME expression coefficients (100-dim)",
    "face_valid": "per-frame: was a face tracked",
    "gaze_pitch": "gaze pitch, radians (L2CS-Net)",
    "gaze_yaw": "gaze yaw, radians (L2CS-Net)",
    "gaze_valid": "per-frame: was gaze estimated",
    "blink_left": "left eye aspect ratio (lower = more closed)",
    "blink_right": "right eye aspect ratio",
    "body_valid": "per-frame: body pose passed depth/size sanity checks",
    "K_fullimg": "camera intrinsics per frame",
    "num_frames": "clip length",
    "coord_system": "which frame the primary arrays are in",
    "ik_coverage": "fraction of hand frames the IK could constrain",
    "ik_wrist_loss": "per-frame residual of the wrist fit",
    "ik_n_targets": "number of IK targets used",
}
for k in keys_seen:
    sh, dt = shapes[k]
    w(f"| `{k}` | {sh} | {dt} | {MEAN.get(k, '')} |")
w("")
w("`*_valid` masks matter: hands are frequently occluded or out of frame. Where a")
w("hand is unobserved the pose is interpolated (SLERP) or falls back to a relaxed")
w("MANO pose, so **filter on the mask** rather than assuming every frame is measured.")
w("")

w("## How it was produced")
w("")
w("Per clip, `vid2smplx/scripts/process_video.py` runs:")
w("")
w("| stage | tool | output |")
w("|---|---|---|")
w("| 1. Body | **GVHMR** (SIGGRAPH Asia 2024) — YOLOv8 tracking → ViTPose 2D keypoints "
  "→ HMR2 ViT features → HMR4D sequence transformer | world- and camera-frame SMPL body |")
w("| 2. Hands | **HaMeR** (CVPR 2024) ViT-H, with a batched whole-body ViTPose hand detector "
  "| MANO parameters per detected hand |")
w("| 3. Face | **EMICA** (INFERNO) | FLAME expression, jaw, eye pose |")
w("| 4. Gaze/blink | **L2CS-Net** + **MediaPipe** FaceMesh eye-aspect-ratio | gaze pitch/yaw, blink |")
w("| 5. Merge | `merge_body_hands.py` | one SMPL-X parameter set, MANO→SMPL-X hand conversion |")
w("| 6. IK | `ik_hands.py` | arms re-optimised so wrists meet the HaMeR hands, "
  "with signed anti-penetration so hands rest on the body instead of sinking into it |")
w("")
w("The excerpt is rendered by `render_excerpt.py`: it scores per-frame arm+hand joint")
w("velocity, picks the highest-motion 60 s window, and renders the original video beside")
w("a pyrender view of the reconstructed mesh. It is a **visual QC artifact only** — it")
w("contains no data that is not already in the .npz.")
w("")
w("Quality gates in `process_video.py` fail a clip outright if either hand is observed")
w("in <5% of frames, or if the IK stage did not complete. `<name>_summary.json` records")
w("the coverage figures and per-stage timings for every clip.")
w("")

w("## Coverage")
w("")
w("| clip | frames | left hand | right hand | face | excerpt |")
w("|---|---:|---:|---:|---:|:---:|")
for c in sorted(clips, key=lambda r: r["name"]):
    w(f"| {c['name']} | {c['frames']:,} | {c['hl']:.0%} | {c['hr']:.0%} | "
      f"{c['face']:.0%} | {'yes' if c['excerpt'] else '—'} |")
w("")

out = f"{R}/README.md"
open(out, "w").write("\n".join(L) + "\n")
print(f"wrote {out} ({len(clips)} clips)")
