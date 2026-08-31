#!/bin/bash
# ============================================================================
# install.sh - One-command setup for vid2smplx
#
# Creates a single conda environment with all dependencies for:
#   - GVHMR (body estimation)
#   - HaMeR (hand estimation)
#   - EMICA/Inferno (face reconstruction)
#   - L2CS-Net (gaze estimation)
#   - MediaPipe (blink detection)
#
# Third-party repos (GVHMR, HaMeR, Inferno) are git submodules.
#
# Usage:
#   bash install.sh                  # full install
#   bash install.sh --skip-models    # skip model downloads
#   bash install.sh --env-only       # env + packages only
#   bash install.sh --force          # remove existing env and recreate
#   bash install.sh --uv             # use uv + .venv instead of conda
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR"

ENV_NAME="${CONDA_ENV:-vid2smplx}"
PYTHON_VERSION="3.10"
SKIP_MODELS=0
ENV_ONLY=0
FORCE=0
USE_UV=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-models) SKIP_MODELS=1; shift ;;
        --env-only)    ENV_ONLY=1; shift ;;
        --force)       FORCE=1; shift ;;
        --uv)          USE_UV=1; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo "  vid2smplx - Installation"
echo "============================================"
echo "  Repo:    $REPO_DIR"
if [ "$USE_UV" -eq 1 ]; then
    echo "  Env:     uv .venv (Python $PYTHON_VERSION)"
    export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"   # torch + CUDA wheels are ~2 GB
    pip_install() { uv pip install --python "$REPO_DIR/.venv/bin/python" "$@"; }
    in_env()      { "$REPO_DIR/.venv/bin/$1" "${@:2}"; }
else
    echo "  Env:     conda '$ENV_NAME' (Python $PYTHON_VERSION)"
    pip_install() { conda run -n "$ENV_NAME" --no-capture-output pip install "$@"; }
    in_env()      { conda run -n "$ENV_NAME" --no-capture-output "$@"; }
fi
echo ""

# ---- Phase 1: Git submodules ----
echo "=== Phase 1/6: Git Submodules ==="

cd "$REPO_DIR"

# Init top-level submodules (GVHMR, hamer, inferno) without recursing
if [ -f GVHMR/setup.py ] && [ -f hamer/setup.py ] && [ -f inferno/setup.py ]; then
    echo "  [OK] Submodules already present"
else
    git submodule update --init GVHMR hamer inferno
    # GVHMR + HaMeR: recurse into their submodules (DPVO, ViTPose, etc.)
    git submodule update --init --recursive GVHMR hamer
fi

# Inferno: only init the external submodules we actually need, skip private gitlab repos
for sub in external/FOCUS external/SwinTransformer external/TDDFA_V2 \
           external/av_hubert external/emonet external/face-parsing.PyTorch; do
    (cd "$REPO_DIR/inferno" && git submodule update --init "$sub" 2>/dev/null || true)
done

echo "  [OK] Submodules initialized"
echo ""

# ---- Phase 2: Environment ----
echo "=== Phase 2/6: Environment ==="

if [ "$USE_UV" -eq 1 ]; then
    command -v uv >/dev/null || { echo "  uv not found: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
    [ "$FORCE" -eq 1 ] && rm -rf "$REPO_DIR/.venv"
    if [ -x "$REPO_DIR/.venv/bin/python" ]; then
        echo "  [OK] .venv already exists"
    else
        uv venv --python "$PYTHON_VERSION" "$REPO_DIR/.venv"
        pip_install pip setuptools wheel   # some deps build with setup.py --no-build-isolation
    fi
    command -v ffmpeg >/dev/null || echo "  [WARN] ffmpeg not on PATH — install it (apt install ffmpeg)"
else
    if [ "$FORCE" -eq 1 ]; then
        echo "  [--force] Removing existing environment..."
        conda env remove -n "$ENV_NAME" -y 2>/dev/null || true
    fi
    if conda env list | grep -q "^${ENV_NAME} "; then
        echo "  [OK] Environment '$ENV_NAME' already exists"
    else
        echo "  Creating conda environment: $ENV_NAME"
        conda create -n "$ENV_NAME" python="$PYTHON_VERSION" ffmpeg -c conda-forge -y
    fi
fi
echo ""

# ---- Phase 3: Pip packages ----
echo "=== Phase 3/6: Pip Packages ==="

echo "  Installing PyTorch 2.3.0 + CUDA 12.1..."
pip_install \
    torch==2.3.0+cu121 torchvision==0.18.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

echo "  Installing pytorch3d..."
pip_install \
    "pytorch3d @ https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt230/pytorch3d-0.7.6-cp310-cp310-linux_x86_64.whl"

echo "  Installing core dependencies..."
pip_install \
    numpy==1.23.5 \
    opencv-python \
    smplx==0.1.28 \
    trimesh \
    einops \
    timm==0.9.12 \
    lightning==2.3.0 \
    hydra-core==1.3 \
    hydra-zen \
    hydra_colorlog \
    rich \
    scikit-image \
    imageio==2.34.1 \
    av==13.0.0 \
    ffmpeg-python \
    tensorboardX \
    matplotlib \
    termcolor \
    tqdm \
    gdown

echo "  Installing chumpy..."
pip_install --no-build-isolation \
    "chumpy @ git+https://github.com/mattloper/chumpy"

echo "  Installing GVHMR-specific dependencies..."
pip_install \
    ultralytics==8.2.42 \
    cython_bbox \
    lapx \
    wis3d \
    pycolmap

echo "  Installing HaMeR-specific dependencies..."
pip_install \
    pyrender \
    yacs \
    xtcocotools \
    pandas \
    webdataset

echo "  Installing detectron2..."
pip_install --no-build-isolation \
    "detectron2 @ git+https://github.com/facebookresearch/detectron2"

echo "  Installing EMICA/Inferno dependencies..."
pip_install --no-deps --no-build-isolation \
    insightface==0.6.2
pip_install \
    "numpy==1.23.5" onnx onnxruntime-gpu prettytable scikit-learn easydict   # numpy pinned: newer onnx pulls numpy 2, breaking insightface/ultralytics

pip_install --no-deps \
    face-alignment==1.3.5 \
    facenet-pytorch==2.5.2 \
    kornia==0.6.5 \
    albumentations==1.0.3 \
    mediapipe \
    munch \
    compress-pickle \
    hickle \
    decord
pip_install numba

pip_install \
    "transformers<5" \
    huggingface-hub

echo "  Installing L2CS-Net (gaze estimation)..."
pip_install \
    "git+https://github.com/edavalosanaya/L2CS-Net.git@main"

# Safety net: anything above that dragged numpy to 2.x breaks insightface, chumpy and ultralytics
pip_install "numpy==1.23.5"

echo "  [OK] All pip packages installed"
echo ""

# ---- Phase 4: Editable installs (always runs) ----
echo "=== Phase 4/6: Editable Installs ==="

# hamer pins mmcv==1.3.9 whose setup.py imports pkg_resources (removed in setuptools 71)
pip_install "setuptools<71"

echo "  Installing GVHMR..."
pip_install -e "$REPO_DIR/GVHMR"

echo "  Installing HaMeR..."
pip_install --no-build-isolation -e "$REPO_DIR/hamer"

echo "  Installing Inferno..."
pip_install --no-deps --no-build-isolation -e "$REPO_DIR/inferno"

echo "  Installing vid2smplx CLI..."
pip_install --no-deps -e "$REPO_DIR"

echo "  [OK] Editable installs complete"
echo ""

# ---- Phase 5: Model weights ----
if [ "$ENV_ONLY" -eq 1 ]; then
    echo "=== Phase 5/6: Model Downloads (SKIPPED - --env-only) ==="
elif [ "$SKIP_MODELS" -eq 1 ]; then
    echo "=== Phase 5/6: Model Downloads (SKIPPED - --skip-models) ==="
else
    echo "=== Phase 5/6: Model Downloads ==="
    in_env vid2smplx download
fi
echo ""

# ---- Phase 6: Verify ----
echo "=== Phase 6/6: Verification (vid2smplx doctor) ==="
# Import/CUDA breakage fails the install (set -e); missing manual weights (SMPL-X/MANO) do not.
in_env python -c "
from vid2smplx.checks import _check_env
import sys
rows = _check_env('')
bad = [r for r in rows if r[0] == 'MISS']
for _, name, note in bad: print(f'  [FAIL] {name}: {note}')
sys.exit(1 if bad else 0)
"
in_env vid2smplx doctor || true

echo ""
echo "============================================"
echo "  vid2smplx installation complete!"
echo "============================================"
echo ""
echo "  If doctor reported [MISS] items, fix them (see docs/models.md), then:"
[ "$USE_UV" -eq 1 ] && echo "    source .venv/bin/activate" || echo "    conda activate $ENV_NAME"
echo "    vid2smplx doctor"
echo "    vid2smplx run examples/clip_talking.mp4 --percent 10 --final-incam"
echo ""
