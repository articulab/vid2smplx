#!/usr/bin/env python3
"""Add the missing `fps` key to params produced before it was recorded.

Frame index is only a time base if fps travels with the params. This dataset
mixes 25 and 29.97 fps, so a consumer that assumes one rate silently mistimes
the other by ~20%.

    python backfill_fps.py --out <params tree> --src <video tree> [--apply]

Matches each <name>.npz to <name>.mp4 by stem, anywhere under --src. Idempotent:
clips that already carry a non-zero fps are skipped.
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np


def probe_fps(video: Path) -> float | None:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "json", str(video)],
        capture_output=True, text=True)
    try:
        rate = json.loads(r.stdout)["streams"][0]["r_frame_rate"]
        num, den = rate.split("/")
        return float(num) / float(den) if float(den) else None
    except Exception:
        return None


ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.expanduser("~/staging/interpersonality_out"))
ap.add_argument("--src", default=os.path.expanduser("~/staging/interpersonality"))
ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
args = ap.parse_args()

videos = {p.stem: p for p in Path(args.src).rglob("*.mp4")}
npzs = sorted(Path(args.out).rglob("*.npz"))
print(f"{len(npzs)} params, {len(videos)} source videos\n")

todo, skipped, missing, rates = [], 0, [], {}
for f in npzs:
    z = np.load(f, allow_pickle=True)
    if "fps" in z and float(z["fps"]) > 0:
        skipped += 1
        continue
    v = videos.get(f.stem)
    if v is None:
        missing.append(f.stem)
        continue
    fps = probe_fps(v)
    if fps is None:
        missing.append(f.stem)
        continue
    todo.append((f, fps))
    rates[round(fps, 2)] = rates.get(round(fps, 2), 0) + 1

print(f"to backfill: {len(todo)}   already have fps: {skipped}   unmatched: {len(missing)}")
print(f"rates found: {rates}")
if missing:
    print(f"  unmatched: {', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}")

if not args.apply:
    print("\nDRY RUN — rerun with --apply")
    raise SystemExit

for i, (f, fps) in enumerate(todo, 1):
    d = dict(np.load(f, allow_pickle=True))
    d["fps"] = np.float32(fps)
    # np.savez appends .npz unless the name already ends in it, so name the
    # temp file accordingly or os.replace looks for a file that was never written.
    tmp = f.with_name(f.name + ".tmp.npz")
    np.savez_compressed(tmp, **d)
    os.replace(tmp, f)          # atomic on the same filesystem
    if i % 20 == 0 or i == len(todo):
        print(f"  {i}/{len(todo)}")
print(f"done: {len(todo)} updated")
