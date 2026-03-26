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
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR"

ENV_NAME="vid2smplx"
PYTHON_VERSION="3.10"
SKIP_MODELS=0
ENV_ONLY=0
FORCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-models) SKIP_MODELS=1; shift ;;
        --env-only)    ENV_ONLY=1; shift ;;
        --force)       FORCE=1; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo "  vid2smplx - Installation"
echo "============================================"
echo "  Repo:    $REPO_DIR"
echo "  Env:     $ENV_NAME (Python $PYTHON_VERSION)"
echo ""

# ---- Phase 1: Git submodules ----
echo "=== Phase 1/6: Git Submodules ==="

cd "$REPO_DIR"

# Init top-level submodules (GVHMR, hamer, inferno) without recursing
git submodule update --init GVHMR hamer inferno

# GVHMR + HaMeR: recurse into their submodules (DPVO, ViTPose, etc.)
git submodule update --init --recursive GVHMR hamer

# Inferno: only init the external submodules we actually need, skip private gitlab repos
for sub in external/FOCUS external/SwinTransformer external/TDDFA_V2 \
           external/av_hubert external/emonet external/face-parsing.PyTorch; do
    (cd "$REPO_DIR/inferno" && git submodule update --init "$sub" 2>/dev/null || true)
done

echo "  [OK] Submodules initialized"
echo ""

# ---- Phase 2: Conda environment ----
echo "=== Phase 2/6: Conda Environment ==="

if [ "$FORCE" -eq 1 ]; then
    echo "  [--force] Removing existing environment..."
    conda env remove -n "$ENV_NAME" -y 2>/dev/null || true
fi

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "  [OK] Environment '$ENV_NAME' already exists"
else
    echo "  Creating conda environment: $ENV_NAME"
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi
echo ""

# ---- Phase 3: Pip packages ----
echo "=== Phase 3/6: Pip Packages ==="

echo "  Installing PyTorch 2.3.0 + CUDA 12.1..."
conda run -n "$ENV_NAME" --no-capture-output pip install \
    torch==2.3.0+cu121 torchvision==0.18.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

echo "  Installing pytorch3d..."
conda run -n "$ENV_NAME" --no-capture-output pip install \
    "pytorch3d @ https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt230/pytorch3d-0.7.6-cp310-cp310-linux_x86_64.whl"

echo "  Installing core dependencies..."
conda run -n "$ENV_NAME" --no-capture-output pip install \
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
conda run -n "$ENV_NAME" --no-capture-output pip install --no-build-isolation \
    "chumpy @ git+https://github.com/mattloper/chumpy"

echo "  Installing GVHMR-specific dependencies..."
conda run -n "$ENV_NAME" --no-capture-output pip install \
    ultralytics==8.2.42 \
    cython_bbox \
    lapx \
    wis3d \
    pycolmap

echo "  Installing HaMeR-specific dependencies..."
conda run -n "$ENV_NAME" --no-capture-output pip install \
    pyrender \
    yacs \
    xtcocotools \
    pandas \
    webdataset

echo "  Installing detectron2..."
conda run -n "$ENV_NAME" --no-capture-output pip install --no-build-isolation \
    "detectron2 @ git+https://github.com/facebookresearch/detectron2"

echo "  Installing EMICA/Inferno dependencies..."
conda run -n "$ENV_NAME" --no-capture-output pip install --no-deps --no-build-isolation \
    insightface==0.6.2
conda run -n "$ENV_NAME" --no-capture-output pip install \
    onnx onnxruntime-gpu prettytable scikit-learn easydict

conda run -n "$ENV_NAME" --no-capture-output pip install --no-deps \
    face-alignment==1.3.5 \
    facenet-pytorch==2.5.2 \
    kornia==0.6.5 \
    albumentations==1.0.3 \
    mediapipe \
    munch \
    compress-pickle \
    hickle \
    decord
conda run -n "$ENV_NAME" --no-capture-output pip install numba

conda run -n "$ENV_NAME" --no-capture-output pip install \
    "transformers<5" \
    huggingface-hub

echo "  Installing L2CS-Net (gaze estimation)..."
conda run -n "$ENV_NAME" --no-capture-output pip install \
    "git+https://github.com/edavalosanaya/L2CS-Net.git@main"

echo "  [OK] All pip packages installed"
echo ""

# ---- Phase 4: Editable installs (always runs) ----
echo "=== Phase 4/6: Editable Installs ==="

echo "  Installing GVHMR..."
conda run -n "$ENV_NAME" --no-capture-output pip install -e "$REPO_DIR/GVHMR"

echo "  Installing HaMeR..."
conda run -n "$ENV_NAME" --no-capture-output pip install --no-build-isolation -e "$REPO_DIR/hamer"

echo "  Installing Inferno..."
conda run -n "$ENV_NAME" --no-capture-output pip install --no-deps --no-build-isolation -e "$REPO_DIR/inferno"

echo "  [OK] Editable installs complete"
echo ""

# ---- Phase 5: Model weights ----
if [ "$ENV_ONLY" -eq 1 ]; then
    echo "=== Phase 5/6: Model Downloads (SKIPPED - --env-only) ==="
elif [ "$SKIP_MODELS" -eq 1 ]; then
    echo "=== Phase 5/6: Model Downloads (SKIPPED - --skip-models) ==="
else
    echo "=== Phase 5/6: Model Downloads ==="
    bash "$REPO_DIR/scripts/download_models.sh"
fi
echo ""

# ---- Phase 6: Verify ----
echo "=== Phase 6/6: Verification ==="

conda run -n "$ENV_NAME" --no-capture-output python -c "
import sys

modules = {
    'torch':        'import torch',
    'pytorch3d':    'import pytorch3d',
    'smplx':        'import smplx',
    'hmr4d (GVHMR)':'import hmr4d',
    'hamer':        'import hamer',
    'inferno':      'import inferno',
    'detectron2':   'import detectron2',
    'ultralytics':  'import ultralytics',
    'insightface':  'import insightface',
    'face_alignment':'import face_alignment',
    'l2cs':         'from l2cs import Pipeline',
    'mediapipe':    'import mediapipe',
    'lightning':    'import lightning',
    'hydra':        'import hydra',
}

failed = []
for name, stmt in modules.items():
    try:
        exec(stmt)
        print(f'  [OK] {name}')
    except Exception as e:
        print(f'  [FAIL] {name}: {e}')
        failed.append(name)

print()
if failed:
    names = ', '.join(failed)
    print(f'  FAILED: {len(failed)} package(s) - {names}')
    sys.exit(1)
else:
    print('  All packages verified successfully!')
" 2>&1 | grep -v -E "pkg_resources is deprecated|DeprecationWarning|FutureWarning"

echo ""
echo "============================================"
echo "  vid2smplx installation complete!"
echo "============================================"
echo ""
echo "  Quick test:"
echo "    bash scripts/process_video.sh /path/to/video.mp4 --percent 5"
echo ""
echo "  Production mode (NPZ only, no renders):"
echo "    bash scripts/process_video.sh /path/to/video.mp4 --production"
echo ""
