"""Readiness checks: every weight/model file the pipeline needs, in one list.

`vid2smplx doctor` prints this table; `vid2smplx download` runs the downloader
and creates the symlinks the submodules expect.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# (group, relative path, how to get it, min bytes). `how` is printed verbatim next to a
# [MISS] row, so it must name a command the reader can actually run -- DOWNLOADABLE is the
# fetched-by-`vid2smplx download` case; the hand-registration ones spell out the site.
# min_bytes catches truncated downloads that exists() passes and torch.load dies on;
# each is a safe floor (~90% of the real artifact), never an exact size.
DOWNLOADABLE = "run `vid2smplx download` (scripts/download_models.sh fetches this)"

MODELS = [
    ("GVHMR", "GVHMR/inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt", DOWNLOADABLE, 140_000_000),
    ("GVHMR", "GVHMR/inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt", DOWNLOADABLE, 2_400_000_000),
    ("GVHMR", "GVHMR/inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth", DOWNLOADABLE, 2_200_000_000),
    ("GVHMR", "GVHMR/inputs/checkpoints/yolo/yolov8x.pt", DOWNLOADABLE, 120_000_000),
    ("GVHMR", "GVHMR/inputs/checkpoints/dpvo/dpvo.pth", DOWNLOADABLE, 12_000_000),
    ("HaMeR", "hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt", DOWNLOADABLE, 2_400_000_000),
    ("HaMeR", "hamer/_DATA/data/mano_mean_params.npz", DOWNLOADABLE, 1_000),
    ("EMICA", "models/inferno/FaceReconstruction/models", DOWNLOADABLE, 2_300_000_000),
    ("EMICA", "models/inferno/mica/model/mica.tar", DOWNLOADABLE, 450_000_000),
    ("EMICA", "inferno/assets/FLAME/geometry/generic_model.pkl",
     DOWNLOADABLE + " -- it lives in EMOCA's FLAME.zip", 47_000_000),
    ("EMICA", "~/.insightface/models/antelopev2/scrfd_10g_bnkps.onnx",
     DOWNLOADABLE + " -- inferno loads insightface from ~/.insightface", 15_000_000),
    ("Hands", "models/mediapipe/hand_landmarker.task", DOWNLOADABLE, 7_000_000),
    ("Gaze", "models/L2CSNet_gaze360.pkl",
     DOWNLOADABLE + " -- HF mirror ymachta/articumotion-checkpoints", 85_000_000),
    ("SMPL-X", "models/smplx/SMPLX_NEUTRAL.npz",
     "register at https://smpl-x.is.tue.mpg.de/ (SMPL-X v1.1 NPZ) -> unzip to models/smplx/", 95_000_000),
    # Ships inside the same SMPL-X v1.1 archive but is easy to miss when only the
    # NPZ zip is extracted; scripts/ik_hands.py hard-fails without it.
    ("SMPL-X", "models/smplx/MANO_SMPLX_vertex_ids.pkl",
     "from the SMPL-X v1.1 archive (models_smplx_v1_1.zip) -> models/smplx/", 12_000),
    ("MANO", "models/mano/MANO_RIGHT.pkl",
     "register at https://mano.is.tue.mpg.de/ (MANO v1.2) -> unzip to models/mano/", 3_400_000),
]

# Submodules hard-code their own asset paths; we point them at models/ with symlinks.
# (group, link location, target) — paths relative to repo root. The group is the same
# key `skip` uses for MODELS: a link into EMICA assets is as skippable as the assets.
LINKS = [
    ("SMPL-X", "GVHMR/inputs/checkpoints/body_models/smplx", "models/smplx"),
    ("MANO", "hamer/_DATA/data/mano", "models/mano"),
    ("EMICA", "inferno/assets/FaceReconstruction", "models/inferno/FaceReconstruction"),
    ("EMICA", "inferno/assets/MICA", "models/inferno/mica"),
]

# The documented FLOOR. Batches are sized from free VRAM (auto_batch.py), so the same clip
# measured 5.1 GB on an 8 GB card and 12.3 GB on a 46 GB one -- the card sets the peak, not the
# clip. 8 GB is the smallest card we have a measured end-to-end success on.
# README and docs/install.md must state this same number.
MIN_VRAM_GB = 8

# What we recommend once a clip runs past everything the small-card evidence covers.
# README, docs/install.md and docs/benchmarks.md must state this same number.
LONG_VIDEO_VRAM_GB = 16

# The longest clip for which we hold a measured, successful GVHMR peak (9.7 GB, rtx8000,
# flat across 727 / 3,587 / 12,526 frames). Past it the only figure we have is the
# pre-`ea4ba35` 13.7 GB at 35,755 frames, which is an UPPER BOUND on a 46 GB card and has
# never been reproduced on a small one -- so past it we recommend, we do not estimate.
MEASURED_TO_FRAMES = 12_526


def recommended_vram_gb(frames: int) -> int:
    """The card size we can stand behind for a clip of `frames` frames.

    Deliberately NOT an estimate of peak usage: peak follows the CARD (every heavy stage sizes
    its batch from free VRAM), so frame count alone cannot predict it. This returns the
    documented recommendation, which is what the user can act on.
    """
    return MIN_VRAM_GB if frames <= MEASURED_TO_FRAMES else LONG_VIDEO_VRAM_GB


def vram_warning(frames: int, available_gb: float) -> str:
    """'' when this card is one we recommend for this clip, else an actionable warning.

    Fires only past the measured range: we have no recorded OOM on any supported card inside it
    (158 corpus run logs, zero OOM lines), and a warning that fires on hardware the docs call
    supported only teaches people to ignore it.
    """
    if not frames or available_gb <= 0:
        return ""
    want = recommended_vram_gb(frames)
    if available_gb >= want * 0.95:
        return ""
    return (
        f"This clip is {frames:,} frames, past the {MEASURED_TO_FRAMES:,} we have measured "
        f"(9.7 GB, rtx8000), and this GPU has {available_gb:.0f} GB. We recommend "
        f"~{want} GB beyond that point: the only longer figure on record is 13.7 GB at 35,755 "
        f"frames on a 46 GB card, and it is an upper bound, not a measurement of what a small "
        f"card needs. Peak VRAM follows the CARD, not the clip length -- every heavy stage "
        f"sizes its batch from free VRAM -- so we cannot predict your peak from frame count.\n"
        f"  If it does run out of memory:\n"
        f"    - use a GPU with at least ~{want} GB, or\n"
        f"    - cut the video into pieces and process them separately.\n"
        f"  GVHMR is the longest stage: on a 20-min video it can run over an hour before "
        f"failing. See docs/benchmarks.md. It is a warning, not a refusal."
    )


IMPORTS = ["torch", "pytorch3d", "smplx", "hmr4d", "hamer", "inferno", "detectron2",
           "ultralytics", "insightface", "l2cs", "mediapipe", "pytorch_lightning"]


def in_env() -> bool:
    """True when the current interpreter already has the pipeline deps (conda activated, or uv .venv)."""
    import importlib.util
    return importlib.util.find_spec("hmr4d") is not None


def make_links(repo: Path = REPO) -> list[str]:
    """Create missing symlinks. Returns list of links created."""
    made = []
    for _group, link, target in LINKS:
        link_p, target_p = repo / link, repo / target
        if link_p.is_symlink():
            continue                      # already a link; leave it alone
        if link_p.is_dir():
            # A submodule tarball can extract an EMPTY dir here, which silently
            # shadows the symlink and hides the real assets (HaMeR's mano).
            # Replace it when empty; a dir with content is deliberate, keep it.
            if any(link_p.iterdir()):
                continue
            link_p.rmdir()
        elif link_p.exists():
            continue
        link_p.parent.mkdir(parents=True, exist_ok=True)
        link_p.symlink_to(target_p)
        made.append(link)
    return made


def _check_env(env: str) -> list[tuple[str, str, str]]:
    """Import every dependency inside the conda env. Returns (status, name, note)."""
    code = ("import importlib,shutil,sys,os.path\n"
            # ffprobe is as load-bearing as ffmpeg (cli.py probes every input) and was
            # unchecked -- doctor passed while `vid2smplx run` died on FileNotFoundError.
            # Also look beside sys.executable: install.sh never activates the env, so a
            # vendored binary in <env>/bin is invisible to shutil.which().
            "for _b in ('ffmpeg','ffprobe'):\n"
            "  _p = shutil.which(_b) or os.path.join(os.path.dirname(sys.executable), _b)\n"
            "  _ok = os.path.exists(_p)\n"
            "  print('OK' if _ok else 'MISS', _b, (_p if not shutil.which(_b) else '') if _ok else 'install it, or re-run install.sh (vendors static-ffmpeg)')\n"
            "for m in sys.argv[1:]:\n"
            "  try: importlib.import_module(m); print('OK', m)\n"
            "  except Exception as e: print('MISS', m, str(e).splitlines()[0][:80])\n"
            "try:\n"
            "  import torch; print('OK' if torch.cuda.is_available() else 'MISS', 'cuda',\n"
            # round, not floor: >>30 showed an 8 GiB card (~7.996) as '7 GB' and WARNed it.
            "    torch.cuda.get_device_name(0)+f' ({round(torch.cuda.get_device_properties(0).total_memory/2**30)} GB)' if torch.cuda.is_available() else 'torch.cuda.is_available() is False')\n"
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


def _vram_row(row: tuple[str, str, str]) -> tuple[str, str, str]:
    """Downgrade an OK cuda row to WARN below MIN_VRAM_GB — it works for short clips, so never fatal."""
    status, name, note = row
    if name != "cuda" or status != "OK":
        return row
    m = re.search(r"\((\d+) GB\)", note)
    if m and int(m.group(1)) < MIN_VRAM_GB:
        return ("WARN", name, f"{note} — below the documented minimum of {MIN_VRAM_GB} GB; "
                              f"expect OOM on long or high-resolution clips, use --percent to shorten")
    return row


def file_status(p: Path, min_bytes: int) -> tuple[str, str]:
    """('OK'|'MISS'|'CORRUPT', note) for one weight file or asset dir, by size not existence."""
    if not p.exists():
        return "MISS", ""
    size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size
    if size < min_bytes:
        return "CORRUPT", (f"only {size} bytes, expected >= {min_bytes} — truncated or failed download "
                           f"(a gdown rate-limit page saves as HTML). Re-run `vid2smplx download`: "
                           f"it re-fetches anything under its floor")
    return "OK", ""


def resolve(rel: str, repo: Path, home: Path | None = None) -> Path:
    """Repo-relative path, or a `~/...` path under `home` (defaults to the real home)."""
    if rel.startswith("~/"):
        return (home or Path.home()) / rel[2:]
    return repo / rel


def doctor(repo: Path = REPO, env: str | None = None, check_env: bool = True, skip: set[str] = frozenset(),
           home: Path | None = None, check_patches: bool = True) -> bool:
    """Print readiness table. Returns True when everything is present. `skip` = model groups not needed."""
    env = env if env is not None else os.environ.get("CONDA_ENV", "vid2smplx")
    rows: list[tuple[str, str, str]] = []
    if check_env:
        rows += _check_env(env)   # imports + ffmpeg + CUDA, evaluated inside the pipeline env
    for group, rel, how, min_bytes in MODELS:
        if group in skip:
            continue
        status, note = file_status(resolve(rel, repo, home), min_bytes)
        rows.append((status, f"{group}: {rel}", note or ("" if status == "OK" else how)))
    for group, link, target in LINKS:
        if group in skip:
            continue
        p = repo / link
        # `exists()` alone passes an empty dir shadowing the link -- that reports
        # OK while the assets are invisible to the submodule. Require content.
        if p.is_symlink():
            ok = p.exists()               # a dangling symlink is not OK
        elif p.is_dir():
            ok = any(p.iterdir())
        else:
            ok = p.exists()
        note = "" if ok else f"run `vid2smplx download` (ln -s {target}); an EMPTY dir here shadows the link"
        rows.append(("OK" if ok else "MISS", f"link: {link}", note))

    if check_patches:
        # An unpatched GVHMR makes the multi-person guard silently inert.
        from .setup_submodules import patch_status
        rows += patch_status(repo)

    rows = [_vram_row(r) for r in rows]
    bad = [r for r in rows if r[0] in ("MISS", "CORRUPT")]
    for status, name, note in rows:
        tag = {"OK": "[OK]  ", "WARN": "[WARN]"}.get(status, f"[{status}]")
        print(f"  {tag} {name}" + (f"\n         -> {note}" if note else ""))
    print()
    if bad:
        print(f"  {len(bad)} item(s) missing or unusable. Most are fetched by `vid2smplx download`; "
              f"the rest need a manual registration and say so. Do what the `->` line under each "
              f"[MISS]/[CORRUPT] row says, then rerun `vid2smplx doctor`.")
    else:
        print("  All checks passed. Try: vid2smplx run examples/clip_talking.mp4 --percent 10 --final-incam")
    return not bad


if __name__ == "__main__":
    sys.exit(0 if doctor() else 1)
