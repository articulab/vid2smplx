"""Readiness checks: every weight/model file the pipeline needs, in one list.

`vid2smplx doctor` prints this table; `vid2smplx download` runs the downloader
and creates the symlinks the submodules expect.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# (group, relative path, how to get it). "auto" = scripts/download_models.sh fetches it.
MODELS = [
    ("GVHMR", "GVHMR/inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt", "auto"),
    ("GVHMR", "GVHMR/inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt", "auto"),
    ("GVHMR", "GVHMR/inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth", "auto"),
    ("GVHMR", "GVHMR/inputs/checkpoints/yolo/yolov8x.pt", "auto"),
    ("GVHMR", "GVHMR/inputs/checkpoints/dpvo/dpvo.pth", "auto"),
    ("HaMeR", "hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt", "auto"),
    ("HaMeR", "hamer/_DATA/data/mano_mean_params.npz", "auto"),
    ("EMICA", "models/inferno/FaceReconstruction/models", "auto"),
    ("EMICA", "models/inferno/mica/model/mica.tar", "auto"),
    ("EMICA", "inferno/assets/FLAME/geometry/generic_model.pkl", "auto (EMOCA FLAME.zip)"),
    ("EMICA", "~/.insightface/models/antelopev2/scrfd_10g_bnkps.onnx", "auto (inferno loads insightface from ~/.insightface)"),
    ("Hands", "models/mediapipe/hand_landmarker.task", "auto"),
    ("Gaze", "models/L2CSNet_gaze360.pkl",
     "gdown or manual: https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd -> models/"),
    ("SMPL-X", "models/smplx/SMPLX_NEUTRAL.npz",
     "register at https://smpl-x.is.tue.mpg.de/ (SMPL-X v1.1 NPZ) -> unzip to models/smplx/"),
    ("MANO", "models/mano/MANO_RIGHT.pkl",
     "register at https://mano.is.tue.mpg.de/ (MANO v1.2) -> unzip to models/mano/"),
]

# Submodules hard-code their own asset paths; we point them at models/ with symlinks.
# (link location, target) — both relative to repo root.
LINKS = [
    ("GVHMR/inputs/checkpoints/body_models/smplx", "models/smplx"),
    ("hamer/_DATA/data/mano", "models/mano"),
    ("inferno/assets/FaceReconstruction", "models/inferno/FaceReconstruction"),
    ("inferno/assets/MICA", "models/inferno/mica"),
]

IMPORTS = ["torch", "pytorch3d", "smplx", "hmr4d", "hamer", "inferno", "detectron2",
           "ultralytics", "insightface", "l2cs", "mediapipe", "pytorch_lightning",
           "mmpose", "h5py"]   # both only surfaced as mid-run crashes


def in_env() -> bool:
    """True when the current interpreter already has the pipeline deps (conda activated, or uv .venv)."""
    import importlib.util
    return importlib.util.find_spec("hmr4d") is not None


def make_links(repo: Path = REPO) -> list[str]:
    """Create missing symlinks. Returns list of links created."""
    made = []
    for link, target in LINKS:
        link_p, target_p = repo / link, repo / target
        if link_p.is_symlink() or (link_p.exists() and any(link_p.iterdir())):
            continue
        # hamer_demo_data.tar.gz ships an EMPTY _DATA/data/mano/ — a placeholder, not a target
        if link_p.exists():
            link_p.rmdir()
        link_p.parent.mkdir(parents=True, exist_ok=True)
        link_p.symlink_to(target_p)
        made.append(link)
    return made


def _check_env(env: str) -> list[tuple[str, str, str]]:
    """Import every dependency inside the conda env. Returns (status, name, note)."""
    code = ("import importlib,shutil,sys\n"
            "print('OK' if shutil.which('ffmpeg') else 'MISS', 'ffmpeg', '' if shutil.which('ffmpeg') else 'apt/conda install ffmpeg')\n"
            "for m in sys.argv[1:]:\n"
            "  try: importlib.import_module(m); print('OK', m)\n"
            "  except Exception as e: print('MISS', m, str(e).splitlines()[0][:80])\n"
            "try:\n"
            "  import torch; print('OK' if torch.cuda.is_available() else 'MISS', 'cuda',\n"
            "    torch.cuda.get_device_name(0)+f' ({torch.cuda.get_device_properties(0).total_memory>>30} GB)' if torch.cuda.is_available() else 'torch.cuda.is_available() is False')\n"
            "except Exception as e: print('MISS', 'cuda', str(e)[:80])\n")
    if in_env():
        cmd = [sys.executable, "-W", "ignore", "-c", code, *IMPORTS]
    else:
        cmd = ["conda", "run", "-n", env, "--no-capture-output", "python", "-W", "ignore", "-c", code, *IMPORTS]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 and not r.stdout.strip():
        return [("MISS", "env", f"conda env '{env}' not found (or activate your .venv first): {r.stderr.strip()[:120]}")]
    rows = []
    for line in r.stdout.splitlines():
        parts = line.split(" ", 2)
        if parts[0] in ("OK", "MISS"):
            rows.append((parts[0], parts[1], parts[2] if len(parts) > 2 else ""))
    return rows


def resolve(rel: str, repo: Path, home: Path | None = None) -> Path:
    """Repo-relative path, or a `~/...` path under `home` (defaults to the real home)."""
    if rel.startswith("~/"):
        return (home or Path.home()) / rel[2:]
    return repo / rel


def doctor(repo: Path = REPO, env: str | None = None, check_env: bool = True, skip: set[str] = frozenset(),
           home: Path | None = None) -> bool:
    """Print readiness table. Returns True when everything is present. `skip` = model groups not needed."""
    env = env if env is not None else os.environ.get("CONDA_ENV", "vid2smplx")
    rows: list[tuple[str, str, str]] = []
    if check_env:
        rows += _check_env(env)   # imports + ffmpeg + CUDA, evaluated inside the pipeline env
    for group, rel, how in MODELS:
        if group in skip:
            continue
        ok = resolve(rel, repo, home).exists()
        rows.append(("OK" if ok else "MISS", f"{group}: {rel}", "" if ok else how))
    for link, target in LINKS:
        p, t = repo / link, repo / target
        if not t.exists():
            continue      # weights absent or group skipped — the MODELS rows above say so
        # Not just "path exists": an empty placeholder leaves the submodule without weights
        ok = p.is_dir() and any(p.iterdir())
        rows.append(("OK" if ok else "MISS", f"link: {link}", "" if ok else f"run `vid2smplx download` (ln -s {target})"))

    missing = [r for r in rows if r[0] == "MISS"]
    for status, name, note in rows:
        tag = "[OK]  " if status == "OK" else "[MISS]"
        print(f"  {tag} {name}" + (f"\n         -> {note}" if note else ""))
    print()
    if missing:
        print(f"  {len(missing)} item(s) missing. Fix the lines marked [MISS] above, then rerun `vid2smplx doctor`.")
    else:
        print("  All checks passed. Try: vid2smplx run examples/clip_talking.mp4 --percent 10 --final-incam")
    return not missing


if __name__ == "__main__":
    sys.exit(0 if doctor() else 1)
