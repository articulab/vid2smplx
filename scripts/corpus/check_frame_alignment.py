#!/usr/bin/env python3
"""Sanity check: does each .npz frame count match its source video?

A mismatch means the params are misaligned with the video timeline — which would
silently corrupt any downstream alignment to audio or transcripts.

    python check_frame_alignment.py [--out ~/staging/interpersonality_out]
                                    [--src ~/staging/interpersonality]
"""
import argparse
import glob
import json
import os
import subprocess

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.expanduser("~/staging/interpersonality_out"))
ap.add_argument("--src", default=os.path.expanduser("~/staging/interpersonality"))
args = ap.parse_args()


def probe(path):
    """(nb_frames, fps) from container metadata; nb_frames may be None."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames,r_frame_rate", "-of", "json", path],
        capture_output=True, text=True)
    try:
        s = json.loads(r.stdout)["streams"][0]
    except Exception:
        return None, None
    nb = s.get("nb_frames")
    nb = int(nb) if nb and nb.isdigit() else None
    num, den = (s.get("r_frame_rate") or "0/1").split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    return nb, fps


rows, missing, mismatch = [], [], []
for f in sorted(glob.glob(f"{args.out}/*/*/*.npz")):
    rel = os.path.relpath(f, args.out)
    sess, cond, base = rel.split(os.sep)
    name = base[:-4]
    vid = f"{args.src}/{sess}/{cond}/{name}.mp4"
    if not os.path.exists(vid):
        missing.append(name)
        continue
    L = int(np.load(f, allow_pickle=True)["num_frames"])
    nb, fps = probe(vid)
    rows.append((name, L, nb, fps))
    if nb is not None and nb != L:
        mismatch.append((name, L, nb, nb - L))

print(f"checked {len(rows)} clips  ({len(missing)} source videos not found locally)\n")
if mismatch:
    print(f"{'clip':22s} {'npz':>8s} {'video':>8s} {'delta':>7s}")
    for n, L, nb, d in mismatch:
        print(f"{n:22s} {L:8,d} {nb:8,d} {d:+7d}")
else:
    print("ALL MATCH — every npz frame count equals its source video frame count")

noprobe = [r[0] for r in rows if r[2] is None]
if noprobe:
    print(f"\n{len(noprobe)} clips had no nb_frames in container metadata "
          f"(not checked): {', '.join(noprobe[:5])}")
if missing:
    print(f"\nsource video absent for {len(missing)}: {', '.join(missing[:5])}"
          f"{' ...' if len(missing) > 5 else ''}")

fps_set = sorted({round(r[3], 3) for r in rows if r[3]})
print(f"\nframe rates seen: {fps_set}")
print(f"total frames: {sum(r[1] for r in rows):,}")
