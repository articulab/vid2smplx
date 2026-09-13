"""vid2smplx command line.

    vid2smplx run <video.mp4> [options]   # video -> smplx_params.npz (+ renders)
    vid2smplx render <clip_dir> [...]     # re-render an existing output dir
    vid2smplx doctor                      # check env, weights, symlinks, submodule patches
    vid2smplx setup                       # verify the submodules carry the changes we need
    vid2smplx download                    # fetch auto-downloadable weights + create symlinks
"""
from __future__ import annotations

import argparse
import contextlib
import errno
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from . import VALID_LAYERS
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


STDERR_TAIL_BYTES = 65536   # enough for any traceback; bounded so a chatty child cannot eat RAM


def _tee_stderr(pipe, sink: list[str]) -> None:
    """Forward the child's stderr to ours as it arrives, keeping a bounded tail for diagnosis.

    Read by chunk, not by line: tqdm writes progress with \\r and line iteration would stall.
    """
    fd = pipe.fileno()
    while True:
        block = os.read(fd, 4096)
        if not block:
            break
        text = block.decode("utf-8", "replace")
        sys.stderr.write(text)
        sys.stderr.flush()
        sink.append(text)
        if sum(map(len, sink)) > 2 * STDERR_TAIL_BYTES:
            sink[:] = ["".join(sink)[-STDERR_TAIL_BYTES:]]


def run(cmd: list[str], cwd: str | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a command, streaming its stdout line by line with noise filtered.

    Stderr is teed: the user still sees it live (tqdm), and the tail is kept so a
    non-zero exit can be explained instead of surfacing as a bare CalledProcessError.
    """
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    err: list[str] = []
    t = threading.Thread(target=_tee_stderr, args=(proc.stderr, err), daemon=True)
    t.start()
    lines = []
    for line in proc.stdout:
        lines.append(line)
        if not NOISE_PATTERNS.search(line):
            print(line, end="", flush=True)
    proc.wait()
    t.join(timeout=5)
    stderr = "".join(err)[-STDERR_TAIL_BYTES:]
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output="".join(lines), stderr=stderr)
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout="".join(lines), stderr=stderr)


def capture(cmd: list[str], cwd: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run a short command in the env, capturing both streams. Never streams to the terminal.

    ffprobe's answers are data, not output: routing them through run() printed loose
    numbers ('1080', '1920', '14.76') above the banner on every single invocation.
    """
    if in_env():
        if cmd and cmd[0] == "python":
            cmd = [sys.executable] + cmd[1:]
    else:
        cmd = ["conda", "run", "-n", CONDA_ENV] + cmd
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


# (signature in the child's stderr, the ONE line a user can act on).
# Ordered: first match wins, so put the specific before the general.
CHILD_ERROR_HINTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED"),
     "The GPU ran out of memory. Retry with a smaller HaMeR batch (`--batch-size 16`, default 48) "
     "and/or a shorter clip (`--percent 25`). Close other GPU programs first: `nvidia-smi` shows what is using it."),
    (re.compile(r"No person was detected anywhere in|"
                r"IndexError: list index out of range[\s\S]*?(?:tracker\.py|get_one_track)|"
                r"(?:tracker\.py|get_one_track)[\s\S]*?IndexError: list index out of range"),
     "No person was detected in the video, so there is no body to reconstruct. GVHMR's YOLO tracker needs a "
     "visible person (confidence >= 0.5). Check the clip actually shows one, and trim to a section where they "
     "are in frame: `ffmpeg -i IN.mp4 -ss 00:00:05 -t 10 -c copy OUT.mp4`."),
    (re.compile(r"InvalidDataError|Invalid data found|NAL unit|moov atom not found|"
                r"error while decoding|Invalid NAL unit size"),
     "The video could not be decoded past its header — the file is truncated or corrupt. "
     "Re-encode it first: `ffmpeg -i IN.mp4 -c:v libx264 -pix_fmt yuv420p -an FIXED.mp4`, then run on FIXED.mp4."),
    (re.compile(r"SIGXFSZ|File size limit exceeded"),
     "A file grew past this shell's file-size limit. Raise it with `ulimit -f unlimited` (check with `ulimit -f`), "
     "or point `--output-dir` at a filesystem without a quota."),
    (re.compile(r"No space left on device|ENOSPC"),
     "The disk holding the output directory is full. Free space or pass `--output-dir` on a bigger volume "
     "(`df -h` shows what is left)."),
    (re.compile(r"SIGKILL|Killed|Out of memory: Killed process"),
     "The process was killed, almost always by the kernel's OOM killer running out of SYSTEM RAM (not VRAM). "
     "Use `--percent` to shorten the clip, or run on a machine with more RAM."),
]


def explain_child_failure(exc: subprocess.CalledProcessError) -> str:
    """One actionable line for a failed child process; a generic but still useful one on no match."""
    text = f"{exc.stderr or ''}\n{exc.output or ''}\nexit {exc.returncode}"
    for pattern, hint in CHILD_ERROR_HINTS:
        if pattern.search(text):
            return hint
    # Our own stages raise RuntimeError/ValueError with an already-actionable message;
    # a generic "it exited 1" would throw that away.
    own = re.findall(r"^(?:RuntimeError|ValueError|FileNotFoundError): (.+)$",
                     exc.stderr or "", re.MULTILINE)
    if own:
        return own[-1].strip()
    tool = Path(str(exc.cmd[1] if len(exc.cmd) > 1 and str(exc.cmd[0]).endswith("python") else exc.cmd[0])).name
    return (f"{tool} exited with code {exc.returncode} and no recognised error signature. "
            f"The child's own output is above — the last lines of it usually name the cause. "
            f"Re-run with the same command to reproduce, and report it with those lines if it is not obvious.")


@contextlib.contextmanager
def dir_lock(d: Path):
    """Advisory exclusive lock on an output dir: two runs writing the same dir corrupt each other."""
    d.mkdir(parents=True, exist_ok=True)
    lock = d / ".lock"
    fh = open(lock, "w")
    try:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            sys.exit(f"another vid2smplx is already running in {d} (lock: {lock}).\n"
                     f"Wait for it to finish, or give this run its own `--output-dir`.")
        fh.write(f"{os.getpid()}\n")
        fh.flush()
        yield
    finally:
        fh.close()


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


def ffprobe_field(video: str, field: str, stream: bool = True) -> str:
    """Get a single stream (or, with stream=False, container) field from a video. '' when absent."""
    select = ["-select_streams", "v:0"] if stream else []
    result = capture(
        ["ffprobe", "-v", "error", *select,
         "-show_entries", f"{'stream' if stream else 'format'}={field}", "-of", "csv=p=0", video],
    )
    val = result.stdout.strip().split("\n")[0].strip() if result.stdout.strip() else ""
    return val


def atomic_write(path: str | Path, suffix: str, write) -> Path:
    """Call write(tmp_path), then atomically move it onto `path`. Nothing lands on failure.

    The one atomic-write implementation in the repo; `scripts/utils.py` imports it too.
    SIGKILL skips the `finally`, so stale temps are reaped here, age-gated at 24 h so a
    concurrent run's live temp is never touched. The temp keeps `suffix`: ffmpeg picks its
    muxer from it and np.savez appends ".npz" unless the name already ends in it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for stale in path.parent.glob(f".{path.name}.*"):
        try:
            if time.time() - stale.stat().st_mtime > 86400:
                stale.unlink()
        except OSError:
            pass
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=suffix)
    os.close(fd)
    try:
        write(tmp)                  # raises on failure, so no rename
        os.replace(tmp, path)       # atomic: a reader sees the old file or the whole new one
    finally:
        Path(tmp).unlink(missing_ok=True)
    return path


def ffmpeg_cached(out: Path, cmd_for, key: dict | None = None, label: str = "") -> None:
    """Produce `out` with ffmpeg once, via a unique temp file renamed on success.

    Writing in place let a killed ffmpeg leave a truncated mp4 that every later run reused.
    The temp name is unique per process: a fixed ".part" name let two concurrent runs
    write the same file and both rename it, so one clip's bytes landed under the other's name.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    if key is None:
        if out.exists():
            return
    elif reuse_or_clear(out, key, label or out.name):
        return
    atomic_write(out, out.suffix, lambda tmp: conda_run(cmd_for(tmp)))
    if key is not None:
        stamp_write(out, key)


def probe_video(path: str) -> str:
    """'' when `path` is a decodable video; otherwise an actionable reason why it is not."""
    hint = f"Run `ffprobe {path}` to see what it actually is; the pipeline needs a decodable video file."
    if ffprobe_field(path, "codec_type") != "video":
        return f"{path} has no video stream (empty, truncated, or a non-video file named .mp4). {hint}"
    w, h = ffprobe_field(path, "width"), ffprobe_field(path, "height")
    if not (w.isdigit() and h.isdigit() and int(w) > 0 and int(h) > 0):
        return f"{path} reports no frame size (got {w!r}x{h!r}) — the file is corrupt. {hint}"
    dur = ffprobe_field(path, "duration", stream=False)
    try:
        seconds = float(dur)
    except ValueError:
        return f"{path} has an unreadable duration ({dur!r}) — the container is damaged. {hint}"
    if seconds <= 0:
        return f"{path} is {seconds}s long — there is nothing to process. {hint}"
    # Metadata is not decodability: a truncated mp4 keeps an intact moov atom, so every
    # check above passes and the run dies ~40s later inside GVHMR on InvalidDataError.
    # Decode every frame -- `-frames:v 1` misses it, the first frame of a truncated file
    # is intact. Video only, no encode: ~40x realtime (0.4s for a 15s 1080p clip).
    # ffmpeg still exits 0 on a decode error, so the stderr text is the signal.
    dec = capture(["ffmpeg", "-v", "error", "-i", path, "-map", "0:v:0", "-f", "null", "-"])
    if dec.returncode != 0 or dec.stderr.strip():
        detail = (dec.stderr.strip().splitlines() or ["no detail"])[-1][:160]
        return (f"{path} has valid metadata but its frames cannot be decoded ({detail}) — the file is "
                f"truncated or corrupt. Re-encode it first: "
                f"`ffmpeg -i {path} -c:v libx264 -pix_fmt yuv420p -an FIXED.mp4`, then run on FIXED.mp4.")
    return ""


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
        self.current: str | None = None   # the stage a crash handler names
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
        self.current = name
        self._starts[name] = time.time()
        self._samples = []
        self._stop = threading.Event()
        threading.Thread(target=self._sample, daemon=True).start()

    def end(self, name):
        self.current = None
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


# ---- stage provenance ------------------------------------------------------
# Keying a cached stage on the FILENAME alone meant `cp other.mp4 A.mp4` and re-run
# returned last week's npz as SUCCESS. Every reusable artifact now carries a stamp
# naming the input it was made from and the flags that shaped it; a mismatch is a rebuild.

def video_identity(video: Path) -> dict:
    """Identity of an input file: content-sensitive, cheap on multi-GB videos."""
    st = video.stat()
    return {"name": video.name, "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def artifact_identity(target) -> dict | None:
    """Identity of a file (or legacy params DIRECTORY) a later stage consumes; None if absent.

    Upstream stage keys describe the stage's INPUTS, so two different outputs can share one.
    Deleting gaze_blink.npz and re-running rebuilt it from the same video, leaving gaze_key —
    and therefore merge_key — byte-identical, so merge printed [SKIP] and smplx_params.npz
    kept the OLD gaze while summary.json reported coverage from the new file. A consumer must
    key on the bytes it actually reads.
    """
    if target is None:
        return None
    p = Path(target)
    if p.is_dir():
        return {"dir": p.name,
                "files": sorted((f.name, f.stat().st_size, f.stat().st_mtime_ns)
                                for f in p.iterdir() if f.is_file())}
    if not p.exists():
        return None
    st = p.stat()
    return {"name": p.name, "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def stamp_path(target: Path) -> Path:
    return target.with_name(target.name + ".stamp.json")


def full_key(key: dict) -> dict:
    """A stage key plus the identity of the code that would produce it.

    Applied here rather than at each call site so no stage can forget it: a stamp that
    describes only inputs and flags reuses output from BEFORE a `git pull`.
    """
    from .provenance import code_identity
    return {**key, "code": dict(code_identity(str(REPO_DIR)))}


def stamp_ok(target: Path, key: dict) -> tuple[bool, str]:
    """(reuse?, why not). Reuse only what exists AND provably came from exactly `key`."""
    if not target.exists():
        return False, ""
    sp = stamp_path(target)
    if not sp.exists():
        return False, "no provenance stamp (made by an older vid2smplx, or by hand)"
    try:
        doc = json.loads(sp.read_text())
    except (OSError, ValueError) as e:
        return False, f"unreadable stamp ({type(e).__name__})"
    # Valid JSON that is not an object ("[1,2]", "null", a bare number) used to reach
    # .get() and surface as a raw AttributeError traceback: main() only catches
    # CalledProcessError, so a hand-edited stamp crashed the run instead of rebuilding.
    if not isinstance(doc, dict):
        return False, (f"stamp at {sp} is JSON but not an object (got {type(doc).__name__}) — "
                       f"hand-edited or truncated; delete it and re-run")
    key = full_key(key)
    got = doc.get("key")
    if got != key:
        changed = sorted({k for k in set(got or {}) | set(key) if (got or {}).get(k) != key.get(k)})
        if changed == ["code"]:
            return False, ("the vid2smplx checkout changed since it was made (a `git pull`, a "
                           "submodule bump, or local edits) — rerunning so the output matches the code")
        return False, f"input or flags changed since it was made ({', '.join(changed) or 'differs'})"
    return True, ""


def stamp_write(target: Path, key: dict) -> None:
    stamp_path(target).write_text(
        json.dumps({"key": full_key(key), "written": time.time()}, indent=2, sort_keys=True))


def artifact_intact(target: Path) -> tuple[str, str]:
    """('intact' | 'broken' | 'unknown', why). Cheap: reads the archive index only.

    Three states, not two. The old two-state version returned the SUCCESS value from its
    ImportError path, so "I could not check" was indistinguishable from "the bytes are
    whole" — and it only ever looked at .npz, while the two LARGEST artifacts
    (hmr4d_results.pt, hamer_hands.pt) are .pt and were never checked at all. Measured:
    a zero-byte hamer_hands.pt with a valid stamp was reused as a finished stage.

    'unknown' still reuses (the orchestrator may legitimately run outside the env, and
    refusing to reuse there would redo hours of GPU work), but it SAYS SO instead of
    claiming the file was verified.
    """
    if not target.exists():
        return "broken", "the file is gone"
    if target.stat().st_size == 0:
        # Needs no library at all, and is the shape a killed writer actually leaves.
        return "broken", "the file is empty (0 bytes) — a killed run left it behind"
    suffix = target.suffix
    if suffix == ".npz":
        try:
            import numpy as np
        except ImportError:
            return "unknown", "numpy is not importable here, so the bytes were not verified"
        try:
            np.load(target, allow_pickle=True).files
        except Exception as e:
            return "broken", f"the file is unreadable ({type(e).__name__}) — a killed run left it partial"
        return "intact", ""
    if suffix == ".pt":
        # torch.save writes a zip whose central directory is at the END of the file, so a
        # truncated write cannot be opened — a whole-file read (or a torch import, which the
        # orchestrator may not have) is not needed to catch the case that actually happens.
        import zipfile
        with target.open("rb") as fh:
            magic = fh.read(2)
        if magic != b"PK":
            return "unknown", "not a torch zip archive (legacy pickle format?) — bytes not verified"
        try:
            with zipfile.ZipFile(target) as z:
                z.namelist()
        except Exception as e:
            return "broken", (f"the archive index is unreadable ({type(e).__name__}) — a killed run "
                              f"left it partial")
        return "intact", ""
    return "unknown", f"no integrity check exists for a '{suffix or 'no-extension'}' file"


def reuse_or_clear(target: Path, key: dict, label: str, extra_clear: list[Path] = ()) -> bool:
    """True to skip `label`. On a mismatch, say why and delete the stale artifacts first."""
    ok, why = stamp_ok(target, key)
    unverified = ""
    if ok:
        # A stamp proves provenance, not that the bytes are whole.
        state, detail = artifact_intact(target)
        if state == "broken":
            ok, why = False, detail
        elif state == "unknown":
            unverified = detail
    if ok:
        print(f"  [SKIP] Already exists (same input and flags): {target}")
        if unverified:
            print(f"  [NOTE] integrity NOT checked: {unverified}. "
                  f"If this stage misbehaves, delete {target} and re-run.")
        return True
    if target.exists():
        print(f"  [STALE] Recomputing {label}: {why}")
    for p in (target, stamp_path(target), *extra_clear):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)
    return False


def dir_has_files(path):
    p = Path(path)
    return p.is_dir() and any(p.iterdir())


def _torch_query(code: str, *args: str) -> str:
    """Run a one-liner that needs torch inside the env (the orchestrator may lack it). Last line of stdout."""
    cmd = ["python", "-c", code, *args]
    if not in_env():
        cmd = ["conda", "run", "-n", CONDA_ENV, "--no-capture-output"] + cmd
    else:
        cmd[0] = sys.executable
    return subprocess.check_output(cmd, text=True).strip().splitlines()[-1]


def _extract_focal(gvhmr_result) -> str:
    """Extract focal length from GVHMR result .pt file."""
    return _torch_query(
        "import sys,torch;"
        "print(float(torch.load(sys.argv[1],map_location='cpu',weights_only=False)['K_fullimg'][0][0][0]))",
        str(gvhmr_result))


def frame_count(video: str) -> int:
    """Frames in `video`; 0 when unknown. nb_frames is missing on some containers, so fall
    back to duration x fps."""
    n = ffprobe_field(video, "nb_frames")
    if n.isdigit() and int(n) > 0:
        return int(n)
    try:
        num, den = ffprobe_field(video, "r_frame_rate").split("/")
        return int(float(ffprobe_field(video, "duration", stream=False)) * float(num) / float(den))
    except (ValueError, ZeroDivisionError):
        return 0


def gpu_total_gb() -> float:
    """Total VRAM of the visible GPU in GB; 0.0 when there is none or torch cannot say."""
    try:
        return float(_torch_query(
            "import torch;print(torch.cuda.get_device_properties(0).total_memory/1e9"
            " if torch.cuda.is_available() else 0)"))
    except (subprocess.CalledProcessError, ValueError):
        return 0.0


def track_summary(bbx_pt: Path) -> dict | None:
    """Who GVHMR saw, and who it kept. None when the file predates track recording."""
    if not bbx_pt.exists():
        return None
    try:
        out = _torch_query(
            "import sys,json,torch;"
            "print(json.dumps(torch.load(sys.argv[1],map_location='cpu',weights_only=False)"
            ".get('track_summary')))",
            str(bbx_pt))
        return json.loads(out)
    except (subprocess.CalledProcessError, ValueError):
        return None


# A non-chosen track is a SECOND SUBJECT only if it is co-present with the chosen one for
# at least this fraction of the video AND its median bbox covers at least this fraction of
# the FRAME. Two orthogonal gates:
#   overlap  -- a YOLO id switch re-identifies the SAME person over DISJOINT frames
#               (overlap ~0) and must never fail a batch job, however long it runs.
#   abs area -- a track too small for HaMeR/EMICA/L2CS to latch onto cannot contaminate the
#               result. Deliberately NOT area relative to the chosen track: a real second
#               subject sitting further back measures 0.24x and must still fail.
MULTI_PERSON_OVERLAP_FRAC = 0.05
MULTI_PERSON_MIN_AREA = 0.01


def _track_extent(t: dict, total: int, chosen: bool = False) -> str:
    ov = t.get("n_overlap_frames")
    ov_txt = (f", {ov} co-present ({ov / total:.1%})" if ov is not None and total else "")
    share = t.get("area_share")
    area = t.get("area_median")
    sh_txt = (f", bbox {area:.1%} of frame" if area is not None else "") + \
             (f" ({share:.2f}x the chosen track)" if share is not None else "")
    return (f"rank {t['rank']}{' <- CHOSEN' if chosen else ''} (track id {t.get('track_id', '?')}): "
            f"{t['n_frames']} frames{ov_txt}"
            f"{sh_txt}, median bbox {[round(float(v)) for v in t['bbx_xyxy_median']]}")


def classify_tracks(summary: dict | None) -> tuple[list[dict], list[dict]]:
    """Split the non-chosen tracks into (second subjects, transient blips).

    A track with no overlap/area fields predates the measurement and counts as a subject:
    an unknown must never downgrade itself into a warning.
    """
    if not summary or summary.get("n_tracks", 1) <= 1:
        return [], []
    total = summary.get("n_frames_total") or 0
    chosen = summary.get("chosen_rank", 0)
    subjects, blips = [], []
    for t in summary.get("tracks", []):
        if t["rank"] == chosen:
            continue
        ov, area = t.get("n_overlap_frames"), t.get("area_median")
        if ov is None or area is None or not total:
            subjects.append(t)
        elif ov / total >= MULTI_PERSON_OVERLAP_FRAC and area >= MULTI_PERSON_MIN_AREA:
            subjects.append(t)
        else:
            blips.append(t)
    return subjects, blips


def multi_person_failure(summary: dict | None, person: int | None) -> str:
    """The message for a clip holding a SECOND SUBJECT. '' when there is nothing to say.

    GVHMR keeps a single track and silently drops the rest; HaMeR, EMICA and L2CS each run
    their OWN detector and never see that choice, so with two people in frame the hands, face
    and gaze can be attributed to the other person's body. Coverage stays at 100% throughout.
    """
    subjects, _ = classify_tracks(summary)
    if not subjects:
        return ""
    chosen_rank = summary.get("chosen_rank", 0)
    total = summary.get("n_frames_total") or 0
    chosen_t = next((t for t in summary["tracks"] if t["rank"] == chosen_rank), None)
    rows = "; ".join([_track_extent(chosen_t, total, chosen=True)] if chosen_t else [])
    rows += "".join("; " + _track_extent(t, total) for t in subjects)
    head = (f"{len(subjects) + 1} people are in this video; the body is reconstructed from ONE of them "
            f"(rank {chosen_rank}, largest first). {rows}.")
    tail = ("Hands (HaMeR), face (EMICA) and gaze (L2CS) each run their own detector and do NOT share that "
            "choice, so they may belong to a different person than the body — the result would be one "
            "plausible-looking body assembled from two people.")
    if person is None:
        return (f"{head} {tail} Crop to one person "
                f"(`ffmpeg -i IN.mp4 -vf crop=W:H:X:Y OUT.mp4`), or pass `--person {chosen_rank}` to accept this "
                f"choice and proceed with the caveat above.")
    return f"{head} {tail} You passed --person {person}; body is that track, the rest is NOT guaranteed to match."


def spurious_track_warning(summary: dict | None) -> str:
    """The message for transient extra tracks kept below the second-subject threshold. '' if none.

    Never fatal: a reflection or a 3-frame passer-by must not kill a 96-clip SLURM array.
    """
    _, blips = classify_tracks(summary)
    if not blips:
        return ""
    total = summary.get("n_frames_total") or 0
    rows = "; ".join(_track_extent(t, total) for t in blips)
    return (f"{len(blips)} extra YOLO track(s) were seen and discarded as transient — {rows}. "
            f"Below the second-subject threshold (co-present for >= {MULTI_PERSON_OVERLAP_FRAC:.0%} of frames "
            f"AND a bbox >= {MULTI_PERSON_MIN_AREA:.0%} of the frame), so the run proceeded on "
            f"the dominant track. Check the clip if you did not expect anyone else in frame.")


REPO_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_DIR / "scripts"


def _int_range(lo: int, hi: int | None = None):
    def parse(v: str) -> int:
        n = int(v)
        if n < lo or (hi is not None and n > hi):
            raise argparse.ArgumentTypeError(f"must be {'between %d and %d' % (lo, hi) if hi else '>= %d' % lo}, got {n}")
        return n
    return parse


def _layer_list(value: str) -> str:
    """argparse type for --layers: a comma list, every entry checked against VALID_LAYERS."""
    names = [v.strip() for v in value.split(",") if v.strip()]
    if not names:
        raise argparse.ArgumentTypeError(f"empty; choose from {','.join(VALID_LAYERS)}")
    bad = [n for n in names if n not in VALID_LAYERS]
    if bad:
        raise argparse.ArgumentTypeError(
            f"unknown layer(s): {','.join(bad)}; choose from {','.join(VALID_LAYERS)}")
    return ",".join(names)


DEPRECATED = {  # flags from the pre-CLI shell driver -> what happens now
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
    r.add_argument("--person", type=_int_range(0), default=None,
                   help="which person to reconstruct when several are in frame "
                        "(rank in the area-sorted track list, 0 = largest). Required to proceed on a "
                        "multi-person video; hands/face/gaze still use their own detectors.")
    r.add_argument("--hand-detector", "--hand_detector", default="mediapipe", help=argparse.SUPPRESS)  # legacy no-op
    for flag in ("--production", "--skip_render", "--use_gvhmr_focal"):
        r.add_argument(flag, action="store_true", help=argparse.SUPPRESS)  # legacy no-ops, warned in validate_run_args
    r.add_argument("--seed", type=int, default=None, help="fix RNG seeds in face/gaze/merge/IK steps (GVHMR and HaMeR are deterministic in eval); used by the functional tests")
    r.add_argument("--cleanup", action="store_true", help="delete intermediates, keep smplx_params.npz + gaze_blink/")
    r.add_argument("--face-method", "--face_method", default="emica", choices=["emica"], help=argparse.SUPPRESS)
    r.add_argument("--skip-doctor", action="store_true", help=argparse.SUPPRESS)

    d = sub.add_parser("render", help="re-render layers of an existing output dir")
    d.add_argument("clip_dir")
    d.add_argument("--layers", type=_layer_list, default="final",
                   help=f"comma list of: {','.join(VALID_LAYERS)}")
    d.add_argument("--video", default="", help="source video if gvhmr/ was cleaned up")

    sub.add_parser("doctor", help="check environment, weights, symlinks and submodule patches")
    sub.add_parser("setup", help="verify the git submodule checkouts contain the changes vid2smplx needs")
    sub.add_parser("download", help="download weights and create submodule symlinks")
    return p


def validate_render_args(args, error) -> None:
    """`run` got input validation; `render` never did, so every mistake reached the child."""
    clip = Path(args.clip_dir)
    if not clip.exists():
        error(f"clip dir not found: {clip}\n"
              f"Pass the output directory of a finished run (the one holding smplx_params.npz), "
              f"e.g. output/<video name>.")
    elif not clip.is_dir():
        error(f"{clip} is a file, not a clip directory. Pass the output directory of a run "
              f"(the one holding smplx_params.npz).")
    elif not (clip / "smplx_params.npz").exists() and not (clip / "gvhmr").is_dir():
        error(f"{clip} holds neither smplx_params.npz nor gvhmr/ — it is not a vid2smplx output dir. "
              f"Run `vid2smplx run <video>` first.")
    elif not _render_inputs_ok(clip, args, error):
        return
    if args.video and not Path(args.video).is_file():
        error(f"--video not found: {args.video}")


# Layers that reproject the body and therefore need GVHMR's own result file.
_LAYERS_NEEDING_GVHMR = {"gvhmr", "final", "global"}


def _render_inputs_ok(clip: Path, args, error) -> bool:
    """`run --cleanup` deletes gvhmr/, so `render` on that dir was a dead end.

    validate_render_args accepted the dir (smplx_params.npz is there), then render.py died
    with a bare `ERROR: Video not found:` naming a path inside the directory --cleanup had
    just removed, and never mentioned --video, the one flag that makes it work.
    """
    gv = clip / "gvhmr" / clip.name
    layers = set(args.layers.split(","))
    if not args.video and not (gv / "0_input_video.mp4").exists():
        error(f"{clip} has no source video: {gv / '0_input_video.mp4'} is missing "
              f"(`vid2smplx run --cleanup` deletes gvhmr/). Re-render by pointing at the "
              f"original video:\n"
              f"  vid2smplx render {clip} --layers {args.layers} --video <your video.mp4>")
        return False
    need = sorted(layers & _LAYERS_NEEDING_GVHMR)
    if need and not (gv / "hmr4d_results.pt").exists():
        keep = sorted(set(VALID_LAYERS) - _LAYERS_NEEDING_GVHMR)
        error(f"layer(s) {','.join(need)} need {gv / 'hmr4d_results.pt'}, which is missing "
              f"(`vid2smplx run --cleanup` deletes gvhmr/). Either re-run the pipeline without "
              f"--cleanup, or render only the layers that do not need it: --layers "
              f"{','.join(keep)}")
        return False
    return True


def cmd_render(args) -> None:
    cmd = ["python", str(SCRIPT_DIR / "render.py"), "--clip_dir", args.clip_dir,
           "--layers", args.layers, "--smplx_dir", str(REPO_DIR / "models" / "smplx")]
    if args.video:
        cmd += ["--video", args.video]
    # check=False: render.py already prints a good message and picks its exit code;
    # re-raising it as CalledProcessError only buried that under a traceback.
    sys.exit(conda_run(cmd, check=False).returncode)


def cmd_download() -> None:
    from .checks import make_links
    run(["bash", str(SCRIPT_DIR / "download_models.sh")])
    for link in make_links():
        print(f"  [link] {link}")


def validate_run_args(args, argv: list[str], error) -> None:
    """Reject impossible inputs, warn on deprecated / redundant flags, normalise combinations."""
    if not Path(args.video).is_file():
        error(f"video not found: {args.video}")
    else:
        # Probe before loading any model: the stages below take minutes to reach the
        # first decode, and an unreadable input used to surface as a raw traceback.
        reason = probe_video(args.video)
        if reason:
            error(reason)
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
    """Last line of defence: no child failure reaches the user as a raw CalledProcessError."""
    try:
        _main(argv)
    except subprocess.CalledProcessError as e:
        sys.exit(f"\n[FAIL] {explain_child_failure(e)}")


def _main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "doctor":
        from .checks import doctor
        sys.exit(0 if doctor() else 1)
    if args.cmd == "setup":
        from .setup_submodules import main as setup_main
        sys.exit(setup_main())
    if args.cmd == "download":
        cmd_download()
        return
    if args.cmd == "render":
        validate_render_args(args, parser.error)
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


def hamer_detection_count(hamer_pt: Path) -> tuple[int | None, str]:
    """(instances reconstructed, why there is no number).

    None means HaMeR *failed* rather than "found nothing" — the two are indistinguishable
    from coverage alone, and only the first is a bug. An empty-but-present file means the
    detector genuinely saw no hands (occluded / out of frame), which is fine. "Absent" and
    "present but unreadable" are different faults and must not share one message.
    """
    if not hamer_pt.exists():
        return None, (f"HaMeR produced no output file at {hamer_pt} — the hand stage failed "
                      f"(distinct from 'hands not visible', which yields an empty result). "
                      f"Re-run; if it fails again the HaMeR stage's own error is in the log above.")
    try:
        import torch as _t
        d = _t.load(hamer_pt, map_location="cpu", weights_only=False)
        return int(len(d.get("frame_idx", []))), ""
    except Exception as e:
        return None, (f"HaMeR's output file {hamer_pt} exists but cannot be read "
                      f"({type(e).__name__}: {str(e).splitlines()[0][:120]}) — it is truncated or was "
                      f"written by an incompatible torch. Delete it and re-run.")


def quality_report(npz_path: Path, timings: dict, n_hand_det: int | None = None,
                   hands: bool = True, face: bool = True, hand_failure: str = "",
                   multi_person: str = "") -> dict:
    """Inspect the produced params and decide whether this clip is actually usable.

    Returns a dict written to summary.json; `failures` being non-empty means the
    run must not be marked SUCCESS. `hands`/`face` are the user's --no-hands/--no-face
    intent: a stage the user turned off is SKIPPED, never a failure.
    """
    import numpy as np

    qc: dict = {"timings_s": dict(timings), "warnings": [], "failures": [],
                "stages": {"hands": "on" if hands else "SKIPPED",
                           "face": "on" if face else "SKIPPED"}}
    if multi_person:
        qc["warnings"].append(multi_person)

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
    # Only report coverage for stages that ran: 'hands_left: 0.0' printed next to
    # 'hands: SKIPPED' reads as a hand failure when it is the user's own --no-hands.
    cov = {}
    if hands:
        cov |= {"hands_left": "left_hand_valid", "hands_right": "right_hand_valid"}
    if face:
        cov |= {"face": "face_valid", "gaze": "gaze_valid"}
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
    if not hands:
        pass                                  # --no-hands: no HaMeR output is the point
    elif n_hand_det is None:
        qc["failures"].append(hand_failure or
                              "HaMeR produced no output file — hand stage failed "
                              "(distinct from 'hands not visible', which yields an empty result)")
    else:
        for side in ("hands_left", "hands_right"):
            if qc[side] < MIN_COVERAGE["hands"]:
                qc["warnings"].append(
                    f"{side} coverage {qc[side]:.1%} — hands mostly not visible, pose is "
                    f"fallback/interpolated; filter on {side.replace('hands_', '')}_hand_valid")
    if face:
        for label in ("face", "gaze"):
            if qc[label] < MIN_COVERAGE[label]:
                qc["warnings"].append(f"{label} coverage {qc[label]:.1%} is low")

    # IK absent entirely means the stage crashed (e.g. a missing model file) and the
    # arms are raw GVHMR — the hand-to-body alignment never happened. That is a
    # failure, not a missing nicety, and "key absent" must not read as "fine".
    if not hands:
        pass                                  # IK aligns hands to the body; there are none
    elif "ik_coverage" not in qc:
        qc["failures"].append("ik_coverage absent — IK stage did not complete")
    elif qc["ik_coverage"] < 0.5:
        qc["warnings"].append(f"IK coverage {qc['ik_coverage']:.1%} is low")

    return qc


def cmd_run(args) -> None:
    """Own the markers and the lock; _run_stages does the work.

    Markers were cleared AFTER quality_report, a point no crash reaches, so a SUCCESS from
    last week survived a run that died in Step 1. They are cleared here, first, and any exit
    that is not a clean SUCCESS leaves a FAILED naming the stage -- batch tooling can then
    tell "crashed" from "never attempted", which a markerless dir never allowed.
    """
    output_base = Path(args.output_dir or str(REPO_DIR / "output")).resolve()
    video_name = Path(args.video).stem + (f"_{args.percent}pct" if args.percent < 100 else "")
    output_dir = output_base / video_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with dir_lock(output_dir):
        # summary.json counts as a marker: a crash that left last run's "failures: []"
        # in place is exactly as misleading as the stale SUCCESS beside it.
        for marker in ("SUCCESS", "FAILED", "summary.json"):
            (output_dir / marker).unlink(missing_ok=True)
        timer = Timer()
        try:
            _run_stages(args, output_dir, timer)
        except SystemExit as e:
            if e.code not in (0, None) and not (output_dir / "FAILED").exists():
                (output_dir / "FAILED").write_text(f"{timer.current or 'setup'}: exited {e.code}\n")
            raise
        except subprocess.CalledProcessError as e:
            hint = explain_child_failure(e)
            (output_dir / "FAILED").write_text(f"{timer.current or 'setup'}: {hint}\n")
            sys.stdout.flush()
            print(f"\n  [FAIL] {timer.current or 'setup'} stage failed.\n  {hint}\n"
                  f"  Wrote FAILED to {output_dir}.", file=sys.stderr)
            sys.exit(1)
        except KeyboardInterrupt:
            (output_dir / "FAILED").write_text(f"{timer.current or 'setup'}: INTERRUPTED\n")
            sys.stdout.flush()
            print("\n  [FAIL] interrupted; wrote FAILED. Re-run to resume from the last complete stage.",
                  file=sys.stderr)
            sys.exit(130)
        except BaseException as e:
            (output_dir / "FAILED").write_text(f"{timer.current or 'setup'}: {type(e).__name__}: {e}\n")
            raise


def _run_stages(args, output_dir: Path, timer: Timer) -> None:

    production = not args.full_debug

    if args.seed is not None:
        os.environ["VID2SMPLX_SEED"] = str(args.seed)
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ.setdefault("RENDERER", "nvdr")
    os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning,ignore::FutureWarning"

    script_dir, repo_dir = SCRIPT_DIR, REPO_DIR

    video = Path(args.video).resolve()
    source_id = video_identity(video)      # the identity every cached stage is keyed on
    video_name = video.stem

    output_base = output_dir.parent

    static_cam = not args.dynamic_cam

    # Trim video if --percent < 100
    if args.percent < 100:
        duration = float(ffprobe_field(str(video), "duration", stream=False))
        target_duration = f"{duration * args.percent / 100:.3f}"
        trim_dir = output_base / ".trimmed"
        trim_dir.mkdir(parents=True, exist_ok=True)
        trimmed = trim_dir / f"{video_name}_{args.percent}pct.mp4"
        if not trimmed.exists():
            print(f"[Trim] Cutting first {args.percent}% ({target_duration}s of {duration}s)...")
        ffmpeg_cached(trimmed, lambda dst: ["ffmpeg", "-y", "-i", str(video), "-t", target_duration,
                                            "-c", "copy", dst, "-loglevel", "warning"],
                      key={"src": source_id, "percent": args.percent}, label="trim")
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
            ffmpeg_cached(downscaled, key={"src": video_identity(video)}, label="downscale", cmd_for=lambda dst: [
                "ffmpeg", "-y", "-i", str(video),
                "-vf", "scale='if(gt(iw,ih),1920,-2)':'if(gt(iw,ih),-2,1920)'",
                "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                "-pix_fmt", "yuv420p", dst, "-loglevel", "warning",
            ])
            video = downscaled
        else:
            print(f"[Resolution] {width}x{height} — OK")

    # Setup paths
    output_dir = output_base / video_name
    gvhmr_dir = repo_dir / "GVHMR"
    hamer_dir = repo_dir / "hamer"
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
        print("Hand:       HaMeR")
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

    # Say up front when the clip runs past what this card size has been measured on.
    from .checks import vram_warning
    warning = vram_warning(frame_count(str(video)), gpu_total_gb())
    if warning:
        print(f"  [WARN] {warning}\n")

    # Step 1: GVHMR (body)
    print("==== Step 1/5: GVHMR body estimation ====")
    timer.start("gvhmr")
    gvhmr_result = gvhmr_out / video_name / "hmr4d_results.pt"
    stage_video_id = video_identity(video)
    gvhmr_key = {"video": stage_video_id, "static_cam": static_cam, "person": args.person or 0}

    if not reuse_or_clear(gvhmr_result, gvhmr_key, "GVHMR", extra_clear=[gvhmr_out / video_name]):
        gvhmr_cmd = ["python", "tools/demo/demo.py",
                     "--video", str(video), "--output_root", str(gvhmr_out)]
        if static_cam:
            gvhmr_cmd.append("-s")
        if args.person is not None:
            gvhmr_cmd.extend(["--person", str(args.person)])
        gvhmr_cmd.append("--no_render")
        conda_run(gvhmr_cmd, cwd=str(gvhmr_dir))
        if not gvhmr_result.exists():
            sys.exit(f"  [ERROR] GVHMR produced no output at {gvhmr_result}. Its own error is above; "
                     f"the usual causes are no visible person and a corrupt clip.")
        stamp_write(gvhmr_result, gvhmr_key)

    print(f"  [OK] Body params: {gvhmr_result}")
    timer.end("gvhmr")

    # Who else was in frame? GVHMR keeps one track and drops the rest without a word.
    summary = track_summary(gvhmr_out / video_name / "preprocess" / "bbx.pt")
    if summary is None:
        # No inventory => the guard cannot answer, and "no message" must never read as
        # "one person". Say which of the two causes it is rather than proceeding blind.
        from .setup_submodules import patch_status
        unpatched = [r for r in patch_status() if r[0] != "OK"]
        detail = ("The GVHMR submodule is at the WRONG COMMIT: "
                  + "; ".join(f"{r[1]} [{r[0]}]" for r in unpatched)
                  + ". Run `git submodule update --init GVHMR`, then `vid2smplx setup` to verify."
                  if unpatched else
                  "The GVHMR submodule is patched, so this bbx.pt predates the track inventory — "
                  "delete it and re-run GVHMR (drop --reuse / clear the gvhmr/ dir).")
        sys.exit(f"  [ERROR] GVHMR recorded no track inventory for {video_name}, so it is unknown whether more "
                 f"than one person was in frame. Continuing would report a clean single-person SUCCESS for a "
                 f"two-person clip. {detail}")

    people = multi_person_failure(summary, args.person)
    blips = spurious_track_warning(summary)
    if blips:
        print(f"  [WARN] {blips}")
    if people:
        print(f"  [MULTI-PERSON] {people}")
        if args.person is None:
            (output_dir / "FAILED").write_text(people + "\n")
            sys.exit(2)

    # Extract focal length from GVHMR result
    gvhmr_focal = _extract_focal(gvhmr_result)
    print(f"  [Focal] {gvhmr_focal}")
    print()

    # Step 2: Hand estimation
    hamer_params_pt = hamer_video_out / "rendered" / "hamer_hands.pt"
    hamer_params_legacy = hamer_video_out / "rendered" / "mano_params"
    hamer_params = hamer_params_pt if hamer_params_pt.exists() else hamer_params_legacy

    def hands_present() -> bool:
        """Does THIS run have hands to hand downstream?

        `--no-hands` must win over whatever an earlier run into the same --output-dir
        left on disk. Merge and IK used to gate on the filesystem alone, so re-running
        with --no-hands silently rebuilt the body from a PREVIOUS run's hamer_hands.pt
        while the summary printed `hands: SKIPPED`. That re-run is exactly the remedy
        IK_COVERAGE_LOW tells the user to apply, so the documented fix for bad hands
        produced a body contaminated by those same hands.
        """
        return not args.no_hands and (hamer_params_pt.exists() or dir_has_files(hamer_params_legacy))

    if args.no_hands:
        print("==== Step 2/5: Hand estimation (SKIPPED) ====")
        print()
    else:
        print("==== Step 2/5: Hand estimation (HaMeR) ====")
        timer.start("hamer")

        # --hand-detector is deprecated and ignored (run_hamer_video.py parses and never reads
        # it), so it must NOT sit in the key: passing it would discard hours of HaMeR work.
        hamer_key = {"video": stage_video_id, "downsample": args.downsample,
                     "batch_size": args.batch_size, "focal": gvhmr_focal}
        if reuse_or_clear(hamer_params_pt, hamer_key, "HaMeR", extra_clear=[hamer_video_out]):
            hamer_params = hamer_params_pt
            print(f"  [SKIP] Already exists: {hamer_params}")
        else:
            hamer_cmd = [
                "python", str(script_dir / "run_hamer_video.py"),
                "--video", str(video), "--out_folder", str(hamer_out),
                "--corpus-dir", str(repo_dir),
                "--downsample", str(args.downsample),
                "--batch-size", str(args.batch_size),
                "--focal-length", gvhmr_focal,
                "--no-render",
            ]
            gvhmr_bbx = gvhmr_out / video_name / "preprocess" / "bbx.pt"
            if gvhmr_bbx.exists():
                hamer_cmd.extend(["--gvhmr-bboxes", str(gvhmr_bbx)])
                print(f"  [Reuse] Using GVHMR YOLO bboxes: {gvhmr_bbx}")
            # HaMeR's CACHE_DIR_HAMER is the CWD-relative "./_DATA"; run from the
            # hamer checkout so it finds the data install.sh already placed there
            # (otherwise it re-downloads 5.7 GB on every fresh clone). Mirrors the
            # gvhmr call above, which already passes cwd.
            conda_run(hamer_cmd, cwd=str(hamer_dir))

            hamer_params = hamer_params_pt if hamer_params_pt.exists() else hamer_params_legacy
            if hamer_params_pt.exists():
                stamp_write(hamer_params_pt, hamer_key)

        # A crash is the caller's problem (conda_run raises on rc != 0); reaching here means
        # HaMeR finished. [OK] must therefore never name a file that is not on disk.
        if hands_present():
            print(f"  [OK] Hand params: {hamer_params}")
        else:
            hamer_params = None
            print("  [WARN] HaMeR finished but saved no hand params: no hands were detected in "
                  "this clip. Continuing without hands — the SMPL-X output will have flat hands. "
                  "If you expected hands, render the body to see what HaMeR was given "
                  "(`vid2smplx render <output dir> --layers final`): hands that are out of frame, "
                  "motion-blurred or smaller than ~40 px are the usual cause.")

        if not production and not args.final_incam:
            hands_incam = render_out / "hands_incam.mp4"
            gvhmr_video = gvhmr_out / video_name / "0_input_video.mp4"
            if not gvhmr_video.exists():
                gvhmr_video = video
            if hands_present() and not hands_incam.exists():
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

        emica_key = {"video": stage_video_id}
        if reuse_or_clear(flame_result_path, emica_key, "EMICA", extra_clear=[emica_out / video_name]):
            flame_result = str(flame_result_path)
        else:
            conda_run([
                "python", str(script_dir / "run_emica.py"),
                "--video", str(video), "--out_folder", str(emica_out),
            ], check=False)
            if flame_result_path.exists():
                flame_result = str(flame_result_path)
                stamp_write(flame_result_path, emica_key)
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

        gaze_key = {"video": video_identity(gvhmr_video)}
        if not reuse_or_clear(gaze_blink_result, gaze_key, "Gaze+Blink",
                              extra_clear=[gaze_blink_out / video_name]):
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
            if gaze_blink_result.exists():
                stamp_write(gaze_blink_result, gaze_key)

        if gaze_blink_result.exists():
            print(f"  [OK] Gaze+Blink: {gaze_blink_result}")
        else:
            print("  [WARN] Gaze+Blink finished but produced no output; continuing without it.")
            gaze_blink_result = None
        timer.end("gaze")
        print()
    else:
        print("==== Step 3.5: Gaze + Blink (SKIPPED) ====")
        gaze_blink_result = None
        print()

    # Step 4: Merge -> SMPL-X
    print("==== Step 4/5: Merge body + hands + face -> SMPL-X ====")
    timer.start("merge")

    # Every input merge actually consumes must be in this key, or regenerating that input
    # leaves the key identical, merge prints [SKIP], and smplx_params.npz keeps the OLD
    # values while summary.json reports coverage computed from the new ones. gaze was
    # passed to merge below but missing from the key — the same class of bug this key exists
    # to prevent. Keyed on what is passed, not on what the stage intended to produce.
    merge_key = {"gvhmr": artifact_identity(gvhmr_result),
                 "hamer": artifact_identity(hamer_params) if hands_present() else None,
                 "face": artifact_identity(flame_result or None),
                 "gaze": artifact_identity(gaze_blink_result)}
    if not reuse_or_clear(smplx_out, merge_key, "merge"):
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
        if hands_present():
            merge_cmd.extend(["--hamer_result", str(hamer_params)])
        if flame_result and Path(flame_result).exists():
            merge_cmd.extend(["--flame_result", flame_result])
        if gaze_blink_result and gaze_blink_result.exists():
            merge_cmd.extend(["--gaze_blink_result", str(gaze_blink_result)])
        conda_run(merge_cmd)
        stamp_write(smplx_out, merge_key)

    print(f"  [OK] SMPL-X params: {smplx_out}")
    timer.end("merge")
    print()

    # Step 4.5: IK hands
    if hands_present():
        try:
            import numpy as np
            ik_already = smplx_out.exists() and "ik_wrist_loss" in np.load(str(smplx_out), allow_pickle=True)
        except Exception:     # outside the env, or unreadable: rerunning IK is safe (idempotent)
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
                msg = ("IK_COVERAGE_LOW: too few frames had a usable wrist target to align hands to the "
                       "body. The hands are mostly not visible in this clip — trim to a section where they "
                       "are, or re-run with --no-hands to keep the body only.")
                print(f"  [FAIL] {msg}")
                (output_dir / "FAILED").write_text(msg + "\n")
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
    # Never read a stale hamer_hands.pt the user asked us to ignore: with --no-hands
    # there is no hand stage to report on, and quality_report skips it anyway.
    n_hand_det, hand_failure = ((None, "") if args.no_hands
                                else hamer_detection_count(hamer_params_pt))
    qc = quality_report(smplx_out, timer.timings, n_hand_det=n_hand_det,
                        hands=not args.no_hands, face=bool(args.face_method),
                        hand_failure=hand_failure,
                        multi_person="\n".join(m for m in (people, blips) if m))
    (output_dir / "summary.json").write_text(json.dumps(qc, indent=2))

    print("\n" + "=" * 44)
    print("  QUALITY")
    print("=" * 44)
    pct = {"hands_left", "hands_right", "face", "gaze", "ik_coverage"}   # by key, not by magnitude:
    for k in ("frames", "hands_left", "hands_right", "face", "gaze", "ik_coverage", "npz_mb"):
        if k in qc and qc[k] is not None:                                # a 0.04 MB npz printed as "0.0%"
            v = qc[k]
            print(f"  {k + ':':16s}{v:.1%}" if k in pct else f"  {k + ':':16s}{v}")
    for stage, state in qc["stages"].items():
        if state != "on":
            print(f"  {stage + ':':16s}{state} (disabled on the command line)")
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
        print(f"  [Cleanup] gvhmr/ is gone, so re-rendering needs the original video:\n"
              f"            vid2smplx render {output_dir} --layers hands,face --video {args.video}")

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
