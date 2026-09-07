"""vid2smplx command line.

    vid2smplx run <video.mp4> [options]   # video -> smplx_params.npz (+ renders)
    vid2smplx render <clip_dir> [...]     # re-render an existing output dir
    vid2smplx doctor                      # check env, weights, symlinks
    vid2smplx download                    # fetch auto-downloadable weights + create symlinks
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from .checks import in_env

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
    """Run a command, streaming its stdout line by line with noise filtered. Stderr passes through for tqdm."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, text=True, env=env)
    lines = []
    for line in proc.stdout:
        lines.append(line)
        if not NOISE_PATTERNS.search(line):
            print(line, end="", flush=True)
    proc.wait()
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout="".join(lines), stderr=None)


def conda_run(cmd: list[str], cwd: str | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a command inside the pipeline environment.

    If we are already inside it (conda activate / .venv), run directly with this interpreter;
    otherwise wrap with `conda run -n $CONDA_ENV`.
    """
    if in_env():
        if cmd and cmd[0] == "python":
            cmd = [sys.executable] + cmd[1:]
        return run(cmd, cwd=cwd, check=check)
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


def _nvsmi() -> tuple[int, int] | None:
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                                       "--format=csv,noheader,nounits"], text=True, timeout=2)
        used, util = out.strip().splitlines()[0].split(",")
        return int(used), int(util)
    except Exception:
        return None


class Timer:
    """Per-stage wall time + GPU peak memory / mean utilization (sampled from nvidia-smi every second)."""
    UNDERUTIL = 40  # % — below this the stage is CPU/IO-bound, not GPU-bound

    def __init__(self):
        self.timings = {}
        self._starts = {}
        self.total_start = time.time()
        self._samples: list[tuple[int, int]] = []
        self._stop = None

    def _sample(self):
        while not self._stop.is_set():
            s = _nvsmi()
            if s:
                self._samples.append(s)
            self._stop.wait(1.0)

    def start(self, name):
        self._starts[name] = time.time()
        self._samples = []
        self._stop = threading.Event()
        threading.Thread(target=self._sample, daemon=True).start()

    def end(self, name):
        elapsed = int(time.time() - self._starts[name])
        self.timings[name] = elapsed
        self._stop.set()
        line = f"  [TIMER] {name}: {elapsed}s"
        if self._samples and elapsed >= 5:
            peak = max(u for u, _ in self._samples)
            util = sum(v for _, v in self._samples) / len(self._samples)
            line += f"  [GPU] peak {peak / 1024:.1f} GB, util {util:.0f}%"
            if util < self.UNDERUTIL:
                line += " — GPU underutilized (CPU/IO-bound or small workload); a bigger batch size will not help"
        print(line)

    def total(self):
        return int(time.time() - self.total_start)


def dir_has_files(path):
    p = Path(path)
    return p.is_dir() and any(p.iterdir())


def _extract_focal(gvhmr_result) -> str:
    """Extract focal length from GVHMR result .pt file (inside the env — the orchestrator may lack torch)."""
    code = ("import sys,torch;"
            "print(float(torch.load(sys.argv[1],map_location='cpu',weights_only=False)['K_fullimg'][0][0][0]))")
    cmd = ["python", "-c", code, str(gvhmr_result)]
    if not in_env():
        cmd = ["conda", "run", "-n", CONDA_ENV, "--no-capture-output"] + cmd
    else:
        cmd[0] = sys.executable
    return subprocess.check_output(cmd, text=True).strip().splitlines()[-1]


REPO_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_DIR / "scripts"


def _int_range(lo: int, hi: int | None = None):
    def parse(v: str) -> int:
        n = int(v)
        if n < lo or (hi is not None and n > hi):
            raise argparse.ArgumentTypeError(f"must be {'between %d and %d' % (lo, hi) if hi else '>= %d' % lo}, got {n}")
        return n
    return parse


DEPRECATED = {  # old process_video.sh flags -> what happens now
    "--production": "is the default (no renders)",
    "--skip_render": "is the default (no renders)",
    "--use_gvhmr_focal": "is always on",
    "--hand_detector": "is ignored (HaMeR uses its own detector)",
    "--hand-detector": "is ignored (HaMeR uses its own detector)",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vid2smplx", description="Video -> SMPL-X parameters (body, hands, face, gaze, blink)")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="process a video end-to-end")
    r.add_argument("video", help="input video (mp4)")
    r.add_argument("--output-dir", "--output_dir", default="", help="output root (default: <repo>/output)")
    r.add_argument("--final-incam", "--final_incam", action="store_true", help="render final mesh overlaid on video")
    r.add_argument("--full-debug", "--full_debug", action="store_true", help="render every debug video (body, hands, face, global)")
    r.add_argument("--no-hands", "--no_hands", action="store_true", help="skip HaMeR")
    r.add_argument("--no-face", "--no_face", action="store_true", help="skip EMICA, gaze and blink")
    r.add_argument("--percent", type=_int_range(1, 100), default=100, help="process only the first N%% of the video (testing)")
    r.add_argument("--downsample", type=_int_range(1), default=1, help="run hands on every Nth frame")
    r.add_argument("--batch-size", "--batch_size", type=_int_range(1), default=48, help="HaMeR batch size")
    r.add_argument("--dynamic-cam", "--dynamic_cam", action="store_true", help="moving camera: run visual odometry (default: static)")
    r.add_argument("--hand-detector", "--hand_detector", default="mediapipe", help=argparse.SUPPRESS)  # legacy no-op
    for flag in ("--production", "--skip_render", "--use_gvhmr_focal"):
        r.add_argument(flag, action="store_true", help=argparse.SUPPRESS)  # legacy no-ops, warned in validate_run_args
    r.add_argument("--seed", type=int, default=None, help="fix RNG seeds in face/gaze/merge/IK steps (GVHMR and HaMeR are deterministic in eval); used by the functional tests")
    r.add_argument("--cleanup", action="store_true", help="delete intermediates, keep smplx_params.npz + gaze_blink/")
    r.add_argument("--face-method", "--face_method", default="emica", choices=["emica"], help=argparse.SUPPRESS)
    r.add_argument("--skip-doctor", action="store_true", help=argparse.SUPPRESS)

    d = sub.add_parser("render", help="re-render layers of an existing output dir")
    d.add_argument("clip_dir")
    d.add_argument("--layers", default="final", help="comma list: final,global,hands,face,gvhmr")
    d.add_argument("--video", default="", help="source video if gvhmr/ was cleaned up")

    sub.add_parser("doctor", help="check environment, weights and symlinks")
    sub.add_parser("download", help="download weights and create submodule symlinks")
    return p


def cmd_render(args) -> None:
    cmd = ["python", str(SCRIPT_DIR / "render.py"), "--clip_dir", args.clip_dir,
           "--layers", args.layers, "--smplx_dir", str(REPO_DIR / "models" / "smplx")]
    if args.video:
        cmd += ["--video", args.video]
    conda_run(cmd)


def cmd_download() -> None:
    from .checks import make_links
    run(["bash", str(SCRIPT_DIR / "download_models.sh")])
    for link in make_links():
        print(f"  [link] {link}")


def validate_run_args(args, argv: list[str], error) -> None:
    """Reject impossible inputs, warn on deprecated / redundant flags, normalise combinations."""
    if not Path(args.video).is_file():
        error(f"video not found: {args.video}")
    for flag, note in DEPRECATED.items():
        if flag in argv:
            print(f"  [DEPRECATED] {flag} {note}; remove it from your command.")
    if args.full_debug and args.final_incam:
        print("  [NOTE] --full-debug already includes the final incam render; ignoring --final-incam.")
        args.final_incam = False
    if args.no_hands and args.downsample != 1:
        print("  [NOTE] --downsample has no effect with --no-hands.")
    if args.no_face:
        args.face_method = ""


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "doctor":
        from .checks import doctor
        sys.exit(0 if doctor() else 1)
    if args.cmd == "download":
        cmd_download()
        return
    if args.cmd == "render":
        cmd_render(args)
        return
    validate_run_args(args, argv, parser.error)
    if not args.skip_doctor:
        from .checks import doctor
        skip = set()
        if args.no_face:
            skip |= {"EMICA", "Gaze"}
        if args.no_hands:
            skip |= {"HaMeR", "Hands", "MANO"}
        if not doctor(check_env=False, skip=skip):
            sys.exit(1)
    cmd_run(args)


MIN_COVERAGE = {"hands": 0.30, "face": 0.20, "gaze": 0.20}


def hamer_detection_count(hamer_pt: Path) -> int | None:
    """How many hand instances HaMeR actually reconstructed.

    Returns None if the file is absent, which means HaMeR *failed* rather than
    "found nothing" — the two are indistinguishable from coverage alone, and only
    the first is a bug. An empty-but-present file means the detector genuinely saw
    no hands (subject occluded / out of frame), which is fine.
    """
    if not hamer_pt.exists():
        return None
    try:
        import torch as _t
        d = _t.load(hamer_pt, map_location="cpu", weights_only=False)
        return int(len(d.get("frame_idx", [])))
    except Exception:
        return None


def quality_report(npz_path: Path, timings: dict, n_hand_det: int | None = None) -> dict:
    """Inspect the produced params and decide whether this clip is actually usable.

    Returns a dict written to summary.json; `failures` being non-empty means the
    run must not be marked SUCCESS.
    """
    import numpy as np

    qc: dict = {"timings_s": dict(timings), "warnings": [], "failures": []}

    if not npz_path.exists():
        qc["failures"].append(f"no params written at {npz_path}")
        return qc

    qc["npz_mb"] = round(npz_path.stat().st_size / 1e6, 1)
    try:
        z = np.load(npz_path, allow_pickle=True)
    except Exception as e:  # truncated / corrupt archive
        qc["failures"].append(f"params unreadable: {type(e).__name__}: {e}")
        return qc

    qc["frames"] = int(z["num_frames"]) if "num_frames" in z else 0
    cov = {
        "hands_left": "left_hand_valid", "hands_right": "right_hand_valid",
        "face": "face_valid", "gaze": "gaze_valid",
    }
    for label, key in cov.items():
        qc[label] = float(z[key].mean()) if key in z else 0.0
    if "ik_coverage" in z:
        qc["ik_coverage"] = float(z["ik_coverage"])

    # Low hand coverage has two causes and only one is a bug:
    #
    #   HaMeR failed to run   -> its output file is absent. Whatever the hands were
    #                            doing, we lost them. This is the 59-clip disaster.
    #   Hands not visible     -> file present, few or no detections. The merge fills
    #                            those frames with the MANO mean pose + SLERP, which
    #                            is a fine stand-in, and *_hand_valid records exactly
    #                            which frames were measured. Not a failure.
    #
    # Coverage alone cannot separate these, so we key on HaMeR's own output.
    qc["hand_detections"] = n_hand_det
    if n_hand_det is None:
        qc["failures"].append(
            "HaMeR produced no output file — hand stage failed "
            "(distinct from 'hands not visible', which yields an empty result)")
    else:
        for side in ("hands_left", "hands_right"):
            if qc[side] < MIN_COVERAGE["hands"]:
                qc["warnings"].append(
                    f"{side} coverage {qc[side]:.1%} — hands mostly not visible, pose is "
                    f"fallback/interpolated; filter on {side.replace('hands_', '')}_hand_valid")
    for label in ("face", "gaze"):
        if qc[label] < MIN_COVERAGE[label]:
            qc["warnings"].append(f"{label} coverage {qc[label]:.1%} is low")

    # IK absent entirely means the stage crashed (e.g. a missing model file) and the
    # arms are raw GVHMR — the hand-to-body alignment never happened. That is a
    # failure, not a missing nicety, and "key absent" must not read as "fine".
    if "ik_coverage" not in qc:
        qc["failures"].append("ik_coverage absent — IK stage did not complete")
    elif qc["ik_coverage"] < 0.5:
        qc["warnings"].append(f"IK coverage {qc['ik_coverage']:.1%} is low")

    return qc


def cmd_run(args) -> None:

    # Repair the submodule symlinks first. `download` runs during install, before the
    # registration-gated weights exist, so it links to a target that is not there yet;
    # re-extracting hamer_demo_data.tar.gz then puts an empty dir back in the way.
    # Idempotent and instant, and it turns a mid-run "MANO_RIGHT.pkl does not exist"
    # into nothing at all.
    from .checks import make_links
    for made in make_links():
        print(f"  [link] {made}")

    production = not args.full_debug

    if args.seed is not None:
        os.environ["VID2SMPLX_SEED"] = str(args.seed)
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ.setdefault("RENDERER", "nvdr")
    os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning,ignore::FutureWarning"

    script_dir, repo_dir = SCRIPT_DIR, REPO_DIR

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
        # Frame index is only a time base if fps travels with the params: this
        # dataset mixes 25 and 29.97, so a consumer cannot assume one rate.
        _fps = ffprobe_field(str(video), "r_frame_rate")
        try:
            _num, _den = _fps.split("/")
            merge_cmd.extend(["--fps", str(float(_num) / float(_den))])
        except (ValueError, ZeroDivisionError):
            print(f"  [WARN] could not read fps from {video} (got {_fps!r}); npz will store 0")
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
        try:
            import numpy as np
            ik_already = smplx_out.exists() and "ik_wrist_loss" in np.load(str(smplx_out), allow_pickle=True)
        except ImportError:   # orchestrator outside the env: rerunning IK is safe (idempotent optimization)
            ik_already = False

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

    # ---- Quality gates -------------------------------------------------
    # Every silent failure this pipeline produced looked like success: HaMeR dying
    # still wrote a params file (with zero hands). Coverage is checked against the
    # artifact, not the exit code.
    qc = quality_report(smplx_out, timer.timings,
                        n_hand_det=hamer_detection_count(hamer_params_pt))
    (output_dir / "summary.json").write_text(json.dumps(qc, indent=2))

    print("\n" + "=" * 44)
    print("  QUALITY")
    print("=" * 44)
    for k in ("frames", "hands_left", "hands_right", "face", "gaze", "ik_coverage", "npz_mb"):
        if k in qc and qc[k] is not None:
            v = qc[k]
            print(f"  {k + ':':16s}{v:.1%}" if isinstance(v, float) and v <= 1.0
                  else f"  {k + ':':16s}{v}")
    for w in qc["warnings"]:
        print(f"  [WARN] {w}")
    if qc["failures"]:
        for f in qc["failures"]:
            print(f"  [FAIL] {f}")
        (output_dir / "FAILED").write_text("\n".join(qc["failures"]) + "\n")
        print("\n  Wrote FAILED marker — output is NOT usable.")
        sys.exit(2)

    (output_dir / "SUCCESS").write_text("SUCCESS\n")

    # Cleanup
    if args.cleanup:
        print("  [Cleanup] Removing intermediates...")
        for d in [gvhmr_out, hamer_out, output_dir / "emica"]:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
        print("  [Cleanup] Done - kept smplx_params.npz, gaze_blink/" + (", render/" if render_out.is_dir() else ""))

    # Summary
    print()
    print("=" * 44)
    print("  Done! Output:")
    print("=" * 44)
    print(f"  SMPL-X params: {smplx_out}")
    if gaze_blink_result and gaze_blink_result.exists():
        print(f"  Gaze+Blink:   {gaze_blink_result}")
    if render_out.is_dir():
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
