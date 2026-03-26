#!/usr/bin/env python3
"""Run HaMeR on a video: detect hands and save MANO params.

Usage:
    python scripts/run_hamer_video.py --video input.mp4 --out_folder output/hamer
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_hamer_inference(hamer_dir: str, out_folder: str, batch_size: int = 48,
                        auto_batch_size: bool = True,
                        focal_length: float = 0, video: str = "",
                        img_folder: str = "") -> bool:
    """Run HaMeR on a video or image folder."""
    source = img_folder or video
    print(f"  [HaMeR] Running inference on {source}...")

    cmd = [
        sys.executable, "demo.py",
        "--out_folder", out_folder,
        "--batch_size", str(batch_size),
        *(["--img_folder", img_folder] if img_folder else ["--video", video]),
    ]
    if auto_batch_size:
        cmd.append("--auto_batch_size")
    if focal_length > 0:
        cmd += ["--focal_length", str(focal_length)]

    try:
        subprocess.run(cmd, cwd=hamer_dir, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [HaMeR] Error: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HaMeR on video")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--out_folder", type=str, default="output/hamer")
    parser.add_argument("--corpus-dir", type=str, default=".")
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--max-res", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--auto-batch-size", action="store_true", default=True)
    parser.add_argument("--no-auto-batch-size", action="store_false", dest="auto_batch_size")
    parser.add_argument("--focal-length", type=float, default=0)
    # Legacy args (ignored, kept for backward compat with process_video.py)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--gvhmr-bboxes", type=str, default="")
    parser.add_argument("--hand-detector", type=str, default="vitpose")

    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    video_name = video_path.stem
    corpus_dir = Path(args.corpus_dir).resolve()
    hamer_dir = corpus_dir / "hamer"
    out_folder = Path(args.out_folder).resolve() / video_name
    render_dir = out_folder / "rendered"
    os.makedirs(render_dir, exist_ok=True)

    print(f"=== HaMeR Video Pipeline ===")
    print(f"Video:      {video_path}")
    print(f"Output:     {out_folder}")
    print()

    if args.downsample > 1:
        frames_dir = out_folder / "frames"
        os.makedirs(frames_dir, exist_ok=True)

        vf_filters = [f"select=not(mod(n\\,{args.downsample}))"]
        if args.max_res > 0:
            vf_filters.append(f"scale='if(gt(iw,ih),min({args.max_res},iw),-2)':'if(gt(ih,iw),min({args.max_res},ih),-2)'")

        cmd = ["ffmpeg", "-y", "-i", str(video_path),
               "-vf", ",".join(vf_filters), "-vsync", "vfr",
               "-qscale:v", "2", str(frames_dir / "%06d.jpg")]
        print(f"  [ffmpeg] Extracting frames (downsample={args.downsample})...")
        subprocess.run(cmd, capture_output=True, check=True)

        num_frames = len(list(frames_dir.glob("*.jpg")))
        print(f"  [ffmpeg] Extracted {num_frames} frames")

        success = run_hamer_inference(
            str(hamer_dir), str(render_dir),
            batch_size=args.batch_size,
            auto_batch_size=args.auto_batch_size,
            focal_length=args.focal_length,
            img_folder=str(frames_dir),
        )
        if not success:
            print("HaMeR inference failed!")
            return

        import shutil
        shutil.rmtree(frames_dir)
        print(f"  [Cleanup] Removed {frames_dir}")
    else:
        success = run_hamer_inference(
            str(hamer_dir), str(render_dir),
            batch_size=args.batch_size,
            auto_batch_size=args.auto_batch_size,
            focal_length=args.focal_length,
            video=str(video_path),
        )
        if not success:
            print("HaMeR inference failed!")
            return

    print(f"\n=== Done ===")
    print(f"Output: {render_dir}")


if __name__ == "__main__":
    main()
