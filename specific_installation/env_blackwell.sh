#!/bin/bash
# Create vid2smplx_bw conda env for Blackwell GPUs (sm_120)
# PyTorch 2.10 + CUDA 12.8 + PyTorch3D from source + nvdiffrast
set -e

ENV_NAME=vid2smplx_bw
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Creating $ENV_NAME ==="
conda create -n $ENV_NAME python=3.10 cuda-toolkit=12.8 ffmpeg -c nvidia -c conda-forge -y

echo "=== Installing PyTorch 2.10 + cu128 ==="
conda run -n $ENV_NAME pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128

echo "=== Building PyTorch3D from source (sm_120) ==="
conda run -n $ENV_NAME bash -c "
export CUDA_HOME=\$CONDA_PREFIX
FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST='8.0;8.6;9.0;12.0' \
pip install --no-build-isolation --no-cache-dir 'git+https://github.com/facebookresearch/pytorch3d.git'
"

echo "=== Building nvdiffrast from source ==="
conda run -n $ENV_NAME bash -c "
export CUDA_HOME=\$CONDA_PREFIX
pip install --no-build-isolation --no-cache-dir 'git+https://github.com/NVlabs/nvdiffrast.git'
"

echo "=== Installing pipeline dependencies ==="
# Install deps that won't conflict with torch first
conda run -n $ENV_NAME pip install \
    smplx==0.1.28 \
    ultralytics==8.2.42 \
    "mediapipe==0.10.21" \
    opencv-python \
    "imageio[pyav]" \
    einops trimesh pytorch-lightning \
    facenet-pytorch iopath yacs fvcore \
    --no-deps-for torch torchvision 2>/dev/null || \
conda run -n $ENV_NAME pip install \
    smplx==0.1.28 ultralytics==8.2.42 "mediapipe==0.10.21" \
    opencv-python "imageio[pyav]" einops trimesh \
    pytorch-lightning facenet-pytorch iopath yacs fvcore

# Reinstall correct PyTorch (deps may have downgraded it)
conda run -n $ENV_NAME pip install torch==2.10.0 torchvision==0.25.0 \
    --index-url https://download.pytorch.org/whl/cu128

# l2cs from git (not on PyPI)
conda run -n $ENV_NAME pip install --no-deps \
    "git+https://github.com/Ahmednull/L2CS-Net.git" 2>/dev/null || true

echo "=== Installing local packages (editable) ==="
conda run -n $ENV_NAME pip install -e "$REPO_DIR/GVHMR"
# HaMeR pulls detectron2 which needs CUDA
conda run -n $ENV_NAME bash -c "export CUDA_HOME=\$CONDA_PREFIX && pip install --no-build-isolation -e $REPO_DIR/hamer"
conda run -n $ENV_NAME pip install -e "$REPO_DIR/hamer/third-party/ViTPose"

# INFERNO/EMICA
if [ -d "$REPO_DIR/inferno" ]; then
    conda run -n $ENV_NAME pip install -e "$REPO_DIR/inferno"
fi

echo "=== Verify ==="
conda run -n $ENV_NAME python3 -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Arch: sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}')

import pytorch3d
from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
print(f'PyTorch3D {pytorch3d.__version__} — CUDA rasterize OK')

try:
    import nvdiffrast.torch as dr
    ctx = dr.RasterizeCudaContext()
    print('nvdiffrast — OK')
    del ctx
except Exception as e:
    print(f'nvdiffrast — {e}')

import mediapipe as mp
print(f'MediaPipe {mp.__version__} — solutions: {hasattr(mp, \"solutions\")}')

import smplx, ultralytics, l2cs
print('smplx, ultralytics, l2cs — OK')
"

echo "=== Done ==="
