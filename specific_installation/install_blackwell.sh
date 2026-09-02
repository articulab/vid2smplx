#!/bin/bash
# ============================================================================
# install_blackwell.sh - vid2smplx env for Blackwell GPUs (sm_120)
#
# Creates a separate "vid2smplx_bw" env with PyTorch 2.6+ so CUDA works on
# Blackwell (RTX PRO 1000, etc.). The original "vid2smplx" env (PyTorch 2.3)
# is left untouched for the cluster (RTX 8000 / A100).
#
# Key differences vs install.sh:
#   - PyTorch 2.10.0 + CUDA 12.8 (sm_120 support — 2.6/2.7 lack sm_120 kernels)
#   - pytorch3d built from source (no prebuilt wheel for PyTorch 2.10)
#   - detectron2 built from source
#   - webdataset added (HaMeR dep missing from original)
#   - ffmpeg via conda (not bundled in pip torch)
#   - usercustomize.py patch for torch.load weights_only default
#
# Usage:
#   bash install_blackwell.sh                  # full install
#   bash install_blackwell.sh --skip-models    # skip model downloads
#   bash install_blackwell.sh --env-only       # env + packages only
#   bash install_blackwell.sh --force          # remove existing env and recreate
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR"

ENV_NAME="vid2smplx_bw"
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
echo "  vid2smplx - Blackwell Installation"
echo "============================================"
echo "  Repo:    $REPO_DIR"
echo "  Env:     $ENV_NAME (Python $PYTHON_VERSION)"
echo "  Target:  Blackwell (sm_120) + CUDA 12.8"
echo ""

# ---- Phase 1: Git submodules ----
echo "=== Phase 1/6: Git Submodules ==="

cd "$REPO_DIR"

git submodule update --init GVHMR hamer inferno
git submodule update --init --recursive GVHMR hamer

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

echo "  Installing PyTorch 2.10.0 + CUDA 12.8..."
conda run -n "$ENV_NAME" --no-capture-output pip install \
    torch==2.10.0+cu128 torchvision==0.25.0+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128

echo "  Building pytorch3d from source (this takes ~10-15 min)..."
conda run -n "$ENV_NAME" --no-capture-output pip install --no-build-isolation \
    "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@stable"

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

echo "  Building detectron2 from source..."
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

echo "  Installing ffmpeg..."
conda install -n "$ENV_NAME" -y ffmpeg -c conda-forge

echo "  [OK] All pip packages installed"
echo ""

# ---- Phase 3b: PyTorch compat patches ----
echo "=== Phase 3b: PyTorch Compatibility Patches ==="

# PyTorch 2.10 defaults torch.load(weights_only=True), which breaks older
# checkpoints (YOLO, HaMeR, GVHMR, etc.). Patch it back to False.
SITE_PACKAGES=$(conda run -n "$ENV_NAME" --no-capture-output python -c "import site; print(site.getsitepackages()[0])")
cat > "$SITE_PACKAGES/usercustomize.py" << 'PYEOF'
# vid2smplx Blackwell compat: default weights_only=False for torch.load
# PyTorch 2.10 changed the default to True, breaking older checkpoints
import torch

_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load
PYEOF
echo "  [OK] torch.load weights_only patch installed"
echo ""

# ---- Phase 4: Editable installs ----
echo "=== Phase 4/6: Editable Installs ==="

echo "  Installing GVHMR..."
conda run -n "$ENV_NAME" --no-capture-output pip install -e "$REPO_DIR/GVHMR"

echo "  Installing HaMeR..."
conda run -n "$ENV_NAME" --no-capture-output pip install --no-build-isolation -e "$REPO_DIR/hamer"

echo "  Installing mmpose (ViTPose wholebody, from HaMeR third-party)..."
conda run -n "$ENV_NAME" --no-capture-output pip install -e "$REPO_DIR/hamer/third-party/ViTPose"

echo "  Installing Inferno..."
conda run -n "$ENV_NAME" --no-capture-output pip install --no-deps --no-build-isolation -e "$REPO_DIR/inferno"

echo "  Installing vid2smplx CLI..."
conda run -n "$ENV_NAME" --no-capture-output pip install --no-deps -e "$REPO_DIR"

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

# Blackwell-specific check
import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f'  [GPU] {gpu} (sm_{cap[0]}{cap[1]})')
    # Quick CUDA test
    try:
        x = torch.randn(16, 16, device='cuda')
        y = x @ x.T
        print(f'  [OK] CUDA matmul works')
    except Exception as e:
        print(f'  [FAIL] CUDA matmul: {e}')
        failed.append('cuda_matmul')
else:
    print(f'  [WARN] No CUDA device detected')

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
echo "  vid2smplx Blackwell installation complete!"
echo "============================================"
echo ""
echo "  Activate with: conda activate $ENV_NAME"
echo ""
echo "  Quick test:"
echo "    CONDA_ENV=$ENV_NAME vid2smplx doctor"
echo "    CONDA_ENV=$ENV_NAME vid2smplx run examples/clip_talking.mp4 --percent 10 --final-incam"
echo ""
