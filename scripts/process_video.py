#!/usr/bin/env python3
"""vid2smplx — End-to-end: video -> SMPL-X params + rendered video.

Usage:
    python scripts/process_video.py <video.mp4> [options]

Options:
    --percent N         Process only first N% of video (testing)
    --downsample N      Take every Nth frame for hands (default: 1)
    --batch_size N      HaMeR batch size (default: 48)
    --dynamic_cam       Use visual odometry (default: static)
    --full_debug        Render all debug MP4s
    --final_incam       Render only the final combined incam video
    --output_dir DIR    Custom output directory
    --no_hands          Skip hand estimation
    --no_face           Skip face, gaze, and blink
    --cleanup           Delete intermediates after success
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

CONDA_ENV = os.environ.get("CONDA_ENV", "vid2smplx")

NOISE_PATTERNS = re.compile(
    r"overwriting variable|pkg_resources is deprecated|apex is not installed|"
    r"Fail to import.*MultiScale|not available in reconstructed resnet|copy resnet state dict|"
    r"deprecated pixel format|Processing MICA image|UserWarning: To copy construct|"
    r"UserWarning: You are using a MANO|UserWarning: torch\.(meshgrid|cross|utils\._pytree)|"
    r"Lipreading model not found|No module named .spectre.|Plan failed with a cudnnException|"
    r"state keys that would end up colliding|Lightning automatically upgraded|"
    r"Found keys that are not in the model state|Importing from timm|"
    r"UserWarning: The parameter .pretrained.|Arguments other than a weight enum|"
    r"WARN:.*loadsave.*Unsupported depth|unexpected key in source state_dict|"
    r"do not match exactly|Use load_from_local|UserWarning: torch.cuda.amp|"
    r"sourceTensor.clone|l2cs.*FutureWarning|MediaPipe.*WARNING"
)


def run(cmd: list[str], cwd: str | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a command, filtering noise from stdout. Stderr passes through for tqdm progress."""
    result = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, text=True)
    for line in result.stdout.splitlines():
        if not NOISE_PATTERNS.search(line):
            print(line)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result


def conda_run(cmd: list[str], cwd: str | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a command inside the conda environment."""
    return run(["conda", "run", "-n", CONDA_ENV, "--no-capture-output"] + cmd,
               cwd=cwd, check=check)


def ffprobe_field(video: str, field: str) -> str:
    """Get a single stream field from a video."""
    result = conda_run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", f"stream={field}", "-of", "csv=p=0", video],
        check=False,
    )
    val = result.stdout.strip().split("\n")[0].strip() if result.stdout.strip() else ""
    return val


class Timer:
    def __init__(self):
        self.timings = {}
        self._starts = {}
        self.total_start = time.time()

    def start(self, name):
        self._starts[name] = time.time()

    def end(self, name):
        elapsed = int(time.time() - self._starts[name])
        self.timings[name] = elapsed
        print(f"  [TIMER] {name}: {elapsed}s")

    def total(self):
        return int(time.time() - self.total_start)


def dir_has_files(path):
    p = Path(path)
    return p.is_dir() and any(p.iterdir())


def _extract_focal(gvhmr_result):
    """Extract focal length from GVHMR result .pt file."""
    pred = torch.load(str(gvhmr_result), map_location="cpu", weights_only=False)
    return str(float(pred["K_fullimg"][0][0][0]))


def parse_args():
    parser = argparse.ArgumentParser(description="vid2smplx - Video -> SMPL-X Parameters")
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("--percent", type=int, default=100)
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--dynamic_cam", action="store_true")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--hand_detector", default="vitpose")
    parser.add_argument("--face_method", default="emica")
    parser.add_argument("--no_face", action="store_true")
    parser.add_argument("--full_debug", action="store_true")
    parser.add_argument("--no_hands", action="store_true")
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--final_incam", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.no_face:
        args.face_method = ""
    production = not args.full_debug

    os.environ.setdefault("RENDERER", "nvdr")
    os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning,ignore::FutureWarning"

    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent

    video = Path(args.video).resolve()
    video_name = video.stem

    output_base = Path(args.output_dir or str(repo_dir / "output"))
    output_base.mkdir(parents=True, exist_ok=True)
    output_base = output_base.resolve()

    timer = Timer()
    static_cam = not args.dynamic_cam

    # Trim video if --percent < 100
    if args.percent < 100:
        result = conda_run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(video)], check=False,
        )
        duration = float(result.stdout.strip().split("\n")[0].strip())
        target_duration = f"{duration * args.percent / 100:.3f}"
        trim_dir = output_base / ".trimmed"
        trim_dir.mkdir(parents=True, exist_ok=True)
        trimmed = trim_dir / f"{video_name}_{args.percent}pct.mp4"
        if not trimmed.exists():
            print(f"[Trim] Cutting first {args.percent}% ({target_duration}s of {duration}s)...")
            conda_run(["ffmpeg", "-y", "-i", str(video), "-t", target_duration,
                        "-c", "copy", str(trimmed), "-loglevel", "warning"])
        video = trimmed
        video_name = f"{video_name}_{args.percent}pct"

    # Downscale to 1080p if needed
    width_str = ffprobe_field(str(video), "width")
    height_str = ffprobe_field(str(video), "height")
    if width_str and height_str:
        width, height = int(width_str), int(height_str)
        if max(width, height) > 1920:
            ds_dir = output_base / ".downscaled"
            ds_dir.mkdir(parents=True, exist_ok=True)
            downscaled = ds_dir / f"{video_name}.mp4"
            if not downscaled.exists():
                print(f"[Downscale] {width}x{height} -> 1080p...")
                conda_run([
                    "ffmpeg", "-y", "-i", str(video),
                    "-vf", "scale='if(gt(iw,ih),1920,-2)':'if(gt(iw,ih),-2,1920)'",
                    "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                    "-pix_fmt", "yuv420p", str(downscaled), "-loglevel", "warning",
                ])
            video = downscaled
        else:
            print(f"[Resolution] {width}x{height} — OK")

    # Setup paths
    output_dir = output_base / video_name
    gvhmr_dir = repo_dir / "GVHMR"
    smplx_dir = repo_dir / "models" / "smplx"
    gvhmr_out = output_dir / "gvhmr"
    hamer_out = output_dir / "hamer"
    hamer_video_out = hamer_out / video_name
    render_out = output_dir / "render"
    smplx_out = output_dir / "smplx_params.npz"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Banner
    print("=" * 44)
    print("  vid2smplx - Video -> SMPL-X Pipeline")
    print("=" * 44)
    print(f"Video:      {video}")
    print(f"Name:       {video_name}")
    print(f"Output:     {output_dir}")
    if not args.no_hands:
        print(f"Hand:       HaMeR (detector: {args.hand_detector})")
    else:
        print("Hand:       DISABLED")
    if args.face_method:
        print(f"Face:       {args.face_method}")
    print(f"Camera:     {'static' if static_cam else 'dynamic'}")
    if production:
        print("Mode:       PRODUCTION (no renders)")
    if args.final_incam:
        print("Render:     final incam only")
    print()

    # Step 1: GVHMR (body)
    print("==== Step 1/5: GVHMR body estimation ====")
    timer.start("gvhmr")
    gvhmr_result = gvhmr_out / video_name / "hmr4d_results.pt"

    if gvhmr_result.exists():
        print(f"  [SKIP] Already exists: {gvhmr_result}")
    else:
        gvhmr_cmd = ["python", "tools/demo/demo.py",
                     "--video", str(video), "--output_root", str(gvhmr_out)]
        if static_cam:
            gvhmr_cmd.append("-s")
        gvhmr_cmd.append("--no_render")
        conda_run(gvhmr_cmd, cwd=str(gvhmr_dir))
        if not gvhmr_result.exists():
            print(f"  [ERROR] GVHMR failed - no output at {gvhmr_result}")
            sys.exit(1)

    print(f"  [OK] Body params: {gvhmr_result}")
    timer.end("gvhmr")

    # Extract focal length from GVHMR result
    gvhmr_focal = _extract_focal(gvhmr_result)
    print(f"  [Focal] {gvhmr_focal}")
    print()

    # Step 2: Hand estimation
    hamer_params_pt = hamer_video_out / "rendered" / "hamer_hands.pt"
    hamer_params_legacy = hamer_video_out / "rendered" / "mano_params"
    hamer_params = hamer_params_pt if hamer_params_pt.exists() else hamer_params_legacy

    if args.no_hands:
        print("==== Step 2/5: Hand estimation (SKIPPED) ====")
        print()
    else:
        print(f"==== Step 2/5: Hand estimation (HaMeR + {args.hand_detector}) ====")
        timer.start("hamer")

        if hamer_params_pt.exists() or dir_has_files(hamer_params_legacy):
            print(f"  [SKIP] Already exists: {hamer_params}")
        else:
            hamer_cmd = [
                "python", str(script_dir / "run_hamer_video.py"),
                "--video", str(video), "--out_folder", str(hamer_out),
                "--corpus-dir", str(repo_dir),
                "--downsample", str(args.downsample),
                "--batch-size", str(args.batch_size),
                "--focal-length", gvhmr_focal,
                "--hand-detector", args.hand_detector,
                "--no-render",
            ]
            gvhmr_bbx = gvhmr_out / video_name / "preprocess" / "bbx.pt"
            if gvhmr_bbx.exists():
                hamer_cmd.extend(["--gvhmr-bboxes", str(gvhmr_bbx)])
                print(f"  [Reuse] Using GVHMR YOLO bboxes: {gvhmr_bbx}")
            conda_run(hamer_cmd)

            hamer_params = hamer_params_pt if hamer_params_pt.exists() else hamer_params_legacy
            if not hamer_params_pt.exists() and not dir_has_files(hamer_params_legacy):
                print("  [WARN] No hand params saved (no hands detected?)")

        print(f"  [OK] Hand params: {hamer_params}")

        if not production and not args.final_incam:
            hands_incam = render_out / "hands_incam.mp4"
            gvhmr_video = gvhmr_out / video_name / "0_input_video.mp4"
            if not gvhmr_video.exists():
                gvhmr_video = video
            if (hamer_params_pt.exists() or dir_has_files(hamer_params_legacy)) and not hands_incam.exists():
                print("  [Render] Hands incam...")
                hands_cmd = [
                    "python", str(script_dir / "render.py"),
                    "--clip_dir", str(output_dir),
                    "--layers", "hands",
                    "--smplx_dir", str(smplx_dir),
                ]
                if not gvhmr_video.exists():
                    hands_cmd.extend(["--video", str(video)])
                conda_run(hands_cmd)

        timer.end("hamer")
        print()

    # Step 3: FLAME face tracking (EMICA)
    flame_result = ""
    if args.face_method == "emica":
        print("==== Step 3/5: FLAME face tracking (EMICA) ====")
        timer.start("emica")
        emica_out = output_dir / "emica"
        flame_result_path = emica_out / video_name / "flame_params.npz"

        if flame_result_path.exists():
            print(f"  [SKIP] Already exists: {flame_result_path}")
            flame_result = str(flame_result_path)
        else:
            conda_run([
                "python", str(script_dir / "run_emica.py"),
                "--video", str(video), "--out_folder", str(emica_out),
            ], check=False)
            if flame_result_path.exists():
                flame_result = str(flame_result_path)
            else:
                print("  [WARN] EMICA failed, continuing without face")

        if flame_result:
            print(f"  [OK] FLAME params: {flame_result}")

        if not production and not args.final_incam and flame_result:
            face_incam = render_out / "face_incam.mp4"
            if not face_incam.exists():
                print("  [Render] Face incam...")
                face_cmd = [
                    "python", str(script_dir / "render.py"),
                    "--clip_dir", str(output_dir),
                    "--layers", "face",
                    "--smplx_dir", str(smplx_dir),
                ]
                gvhmr_video = gvhmr_out / video_name / "0_input_video.mp4"
                if not gvhmr_video.exists():
                    face_cmd.extend(["--video", str(video)])
                conda_run(face_cmd)

        timer.end("emica")
        print()
    else:
        print("==== Step 3/5: FLAME face tracking (SKIPPED) ====")
        print()

    # Step 3.5: Gaze + Blink
    gaze_blink_out = output_dir / "gaze_blink"
    gaze_blink_result = gaze_blink_out / video_name / "gaze_blink.npz"

    if args.face_method == "emica":
        print("==== Step 3.5: Gaze + Blink estimation ====")
        timer.start("gaze")

        gvhmr_video = gvhmr_out / video_name / "0_input_video.mp4"
        if not gvhmr_video.exists():
            gvhmr_video = video

        if gaze_blink_result.exists():
            print(f"  [SKIP] Already exists: {gaze_blink_result}")
        else:
            gaze_cmd = [
                "python", str(script_dir / "run_gaze_blink.py"),
                "--video", str(gvhmr_video),
                "--out_folder", str(gaze_blink_out),
                "--corpus_dir", str(repo_dir),
                "--video_name", video_name,
            ]
            emica_cache = output_dir / "emica" / video_name / "_detection_cache.npz"
            if emica_cache.exists():
                gaze_cmd.extend(["--emica_cache", str(emica_cache)])
            conda_run(gaze_cmd)

        print(f"  [OK] Gaze+Blink: {gaze_blink_result}")
        timer.end("gaze")
        print()
    else:
        print("==== Step 3.5: Gaze + Blink (SKIPPED) ====")
        gaze_blink_result = None
        print()

    # Step 4: Merge -> SMPL-X
    print("==== Step 4/5: Merge body + hands + face -> SMPL-X ====")
    timer.start("merge")

    if smplx_out.exists():
        print(f"  [SKIP] Already exists: {smplx_out}")
    else:
        merge_cmd = [
            "python", str(script_dir / "merge_body_hands.py"),
            "--gvhmr_result", str(gvhmr_result),
            "--output", str(smplx_out), "--coord", "global",
        ]
        if hamer_params_pt.exists() or dir_has_files(hamer_params_legacy):
            merge_cmd.extend(["--hamer_result", str(hamer_params)])
        if flame_result and Path(flame_result).exists():
            merge_cmd.extend(["--flame_result", flame_result])
        if gaze_blink_result and gaze_blink_result.exists():
            merge_cmd.extend(["--gaze_blink_result", str(gaze_blink_result)])
        conda_run(merge_cmd)

    print(f"  [OK] SMPL-X params: {smplx_out}")
    timer.end("merge")
    print()

    # Step 4.5: IK hands
    if hamer_params_pt.exists() or dir_has_files(hamer_params_legacy):
        import numpy as np
        ik_already = smplx_out.exists() and "ik_wrist_loss" in np.load(str(smplx_out), allow_pickle=True)

        if ik_already:
            print("==== Step 4.5: IK hands (SKIPPED — already applied) ====")
        else:
            print("==== Step 4.5: IK hands ====")
            timer.start("ik")
            ik_result = conda_run([
                "python", str(script_dir / "ik_hands.py"),
                "--smplx_params", str(smplx_out),
                "--hamer_params", str(hamer_params),
                "--gvhmr_result", str(gvhmr_result),
                "--smplx_dir", str(smplx_dir),
            ], check=False)

            if ik_result.returncode == 77:
                print("  [FAIL] IK coverage too low")
                (output_dir / "FAILED").write_text("IK_COVERAGE_LOW\n")
                sys.exit(1)
            elif ik_result.returncode != 0:
                print(f"  [ERROR] IK failed with exit code {ik_result.returncode}")
            else:
                print(f"  [OK] IK applied to {smplx_out}")
            timer.end("ik")
        print()

    # Step 5: Render
    timer.start("render")
    skip_render = production and not args.final_incam
    if skip_render:
        print("==== Step 5/5: Render (SKIPPED) ====")
    else:
        print("==== Step 5/5: Render body+hands video ====")

        render_cmd = [
            "python", str(script_dir / "render.py"),
            "--clip_dir", str(output_dir),
            "--smplx_dir", str(smplx_dir),
        ]
        if args.final_incam:
            render_cmd.extend(["--layers", "final"])
        else:
            render_cmd.extend(["--layers", "final,global"])
        gvhmr_video = gvhmr_out / video_name / "0_input_video.mp4"
        if not gvhmr_video.exists():
            render_cmd.extend(["--video", str(video)])

        conda_run(render_cmd)
        print(f"  [OK] Rendered: {render_out}")

    timer.end("render")

    (output_dir / "SUCCESS").write_text("SUCCESS\n")

    # Cleanup
    if args.cleanup:
        print("  [Cleanup] Removing intermediates...")
        for d in [gvhmr_out, hamer_out, output_dir / "emica", render_out]:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
        print("  [Cleanup] Done - kept smplx_params.npz + gaze_blink/")

    # Summary
    print()
    print("=" * 44)
    print("  Done! Output:")
    print("=" * 44)
    print(f"  SMPL-X params: {smplx_out}")
    if gaze_blink_result and gaze_blink_result.exists():
        print(f"  Gaze+Blink:   {gaze_blink_result}")
    if not args.cleanup and render_out.is_dir():
        print(f"  Renders:      {render_out}/")
    print()
    print("=" * 44)
    print("  Step Timings:")
    print("=" * 44)
    labels = [
        ("gvhmr", "GVHMR"), ("hamer", "HaMeR"), ("emica", "EMICA"),
        ("gaze", "Gaze+Blink"), ("merge", "Merge"), ("ik", "IK Hands"),
        ("render", "Render"),
    ]
    for key, label in labels:
        if key in timer.timings:
            print(f"  {label + ':':14s}{timer.timings[key]}s")
    print(f"  {'Total:':14s}{timer.total()}s")
    print()


if __name__ == "__main__":
    main()
