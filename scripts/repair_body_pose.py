#!/usr/bin/env python3
"""Repair clips where IK arms landed only in `body_pose_incam`.

ik_hands.py before 2026-04-14 wrote the IK result to `body_pose_incam` only when
coord=="global", leaving `body_pose` holding raw (un-IK'd) GVHMR arms. GVHMR's
body_pose is coord-system invariant (verified: global == incam bit-for-bit), so
the two fields must agree; where they disagree, `body_pose_incam` is the good one.

Idempotent: clips already consistent are skipped.

    python repair_body_pose.py --root output/corpus [--apply] [--jobs 8]
"""
import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ARM_JOINTS = slice(15, 21)  # collar/shoulder/elbow/wrist chain the IK optimizes


def inspect(path):
    """Return (path, status, max_arm_diff)."""
    try:
        z = np.load(path, allow_pickle=True)
    except Exception as e:
        return path, f"unreadable: {e}", 0.0

    if "body_pose" not in z or "body_pose_incam" not in z:
        return path, "skip: missing field", 0.0
    if "ik_wrist_loss" not in z:
        return path, "skip: IK never ran", 0.0

    bp, bpi = z["body_pose"], z["body_pose_incam"]
    if bp.shape != bpi.shape:
        return path, "skip: shape mismatch", 0.0
    if np.allclose(bp, bpi):
        return path, "ok", 0.0

    d = np.abs(bp - bpi).reshape(len(bp), 21, 3)
    non_arm = np.delete(d, np.r_[ARM_JOINTS], axis=1).max()
    if non_arm > 1e-5:
        # Divergence outside the IK'd arm chain means something else is going on;
        # refuse rather than guess.
        return path, f"SKIP: differs outside arms ({non_arm:.4f})", float(d.max())
    return path, "needs-repair", float(d.max())


def repair(path):
    z = dict(np.load(path, allow_pickle=True))
    z["body_pose"] = z["body_pose_incam"].copy()
    tmp = path + ".tmp.npz"
    np.savez(tmp, **z)
    os.replace(tmp, path)  # atomic on same filesystem
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output/corpus")
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.root}/*/smplx_params.npz"))
    print(f"scanning {len(files)} clips under {args.root}")

    with ProcessPoolExecutor(args.jobs) as ex:
        results = list(ex.map(inspect, files))

    todo = [p for p, s, _ in results if s == "needs-repair"]
    counts = {}
    for _, s, _ in results:
        key = s.split(":")[0]
        counts[key] = counts.get(key, 0) + 1
    for k, v in sorted(counts.items()):
        print(f"  {k:32s} {v}")

    worst = sorted([r for r in results if r[1] == "needs-repair"], key=lambda r: -r[2])[:5]
    for p, _, d in worst:
        print(f"  worst: {os.path.basename(os.path.dirname(p))} max_arm_diff={d:.3f} rad")

    if not args.apply:
        print(f"\nDRY RUN — {len(todo)} clips would be repaired. Re-run with --apply.")
        return

    with ProcessPoolExecutor(args.jobs) as ex:
        for i, _ in enumerate(ex.map(repair, todo), 1):
            if i % 250 == 0 or i == len(todo):
                print(f"  repaired {i}/{len(todo)}")
    print(f"done: {len(todo)} clips repaired")


if __name__ == "__main__":
    main()
