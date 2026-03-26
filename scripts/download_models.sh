#!/bin/bash
# ============================================================================
# download_models.sh - Download all model weights for vid2smplx
#
# Usage:
#   bash scripts/download_models.sh          # download all auto-downloadable weights
#   bash scripts/download_models.sh --all    # also attempt SMPL-X/MANO/FLAME (needs manual setup)
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
GVHMR_DIR="$REPO_DIR/GVHMR"
HAMER_DIR="$REPO_DIR/hamer"

DOWNLOAD_ALL=0
[ "$1" = "--all" ] && DOWNLOAD_ALL=1

echo "============================================"
echo "  vid2smplx - Model Downloads"
echo "============================================"
echo ""

# ---- Helper ----
download_if_missing() {
    local url="$1"
    local dest="$2"
    if [ -f "$dest" ]; then
        echo "  [SKIP] Already exists: $dest"
    else
        echo "  [Download] $dest"
        mkdir -p "$(dirname "$dest")"
        wget -q --show-progress -O "$dest" "$url"
    fi
}

# ---- 1. GVHMR checkpoints ----
echo "=== GVHMR Checkpoints ==="
GVHMR_CKPT_DIR="$GVHMR_DIR/inputs/checkpoints"

download_if_missing \
    "https://huggingface.co/camenduru/GVHMR/resolve/main/gvhmr/gvhmr_siga24_release.ckpt" \
    "$GVHMR_CKPT_DIR/gvhmr/gvhmr_siga24_release.ckpt"

download_if_missing \
    "https://huggingface.co/camenduru/GVHMR/resolve/main/hmr2/epoch%3D10-step%3D25000.ckpt" \
    "$GVHMR_CKPT_DIR/hmr2/epoch=10-step=25000.ckpt"

download_if_missing \
    "https://huggingface.co/camenduru/GVHMR/resolve/main/vitpose/vitpose-h-multi-coco.pth" \
    "$GVHMR_CKPT_DIR/vitpose/vitpose-h-multi-coco.pth"

download_if_missing \
    "https://huggingface.co/camenduru/GVHMR/resolve/main/yolo/yolov8x.pt" \
    "$GVHMR_CKPT_DIR/yolo/yolov8x.pt"

download_if_missing \
    "https://huggingface.co/camenduru/GVHMR/resolve/main/dpvo/dpvo.pth" \
    "$GVHMR_CKPT_DIR/dpvo/dpvo.pth"

echo ""

# ---- 2. HaMeR checkpoint ----
echo "=== HaMeR Checkpoint ==="
HAMER_DATA="$HAMER_DIR/_DATA"

if [ -f "$HAMER_DATA/hamer_ckpts/checkpoints/hamer.ckpt" ]; then
    echo "  [SKIP] Already exists: hamer.ckpt"
else
    echo "  [Download] HaMeR demo data..."
    mkdir -p "$HAMER_DATA"
    wget -q --show-progress -O "/tmp/hamer_demo_data.tar.gz" \
        "https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz"
    tar -xzf "/tmp/hamer_demo_data.tar.gz" -C "$HAMER_DATA"
    rm -f "/tmp/hamer_demo_data.tar.gz"
fi
echo ""

# ---- 3. EMICA / Inferno weights ----
echo "=== EMICA / Inferno Weights ==="
INFERNO_ASSETS="$REPO_DIR/models/inferno"

# FaceReconstruction models
if [ -d "$INFERNO_ASSETS/FaceReconstruction" ]; then
    echo "  [SKIP] FaceReconstruction already exists"
else
    echo "  [Download] FaceReconstruction models..."
    mkdir -p "$INFERNO_ASSETS"
    wget -q --show-progress -O "/tmp/FaceReconstruction.zip" \
        "https://download.is.tue.mpg.de/emote/FaceReconstruction.zip"
    unzip -q "/tmp/FaceReconstruction.zip" -d "$INFERNO_ASSETS/"
    rm -f "/tmp/FaceReconstruction.zip"
fi

# MICA model
if [ -f "$INFERNO_ASSETS/mica/mica.tar" ] || [ -d "$INFERNO_ASSETS/mica" ]; then
    echo "  [SKIP] MICA model already exists"
else
    echo "  [Download] MICA model..."
    mkdir -p "$INFERNO_ASSETS/mica"
    wget -q --show-progress -O "$INFERNO_ASSETS/mica/mica.tar" \
        "https://keeper.mpdl.mpg.de/f/db172dc4bd4f4c0f96de/?dl=1"
    tar -xf "$INFERNO_ASSETS/mica/mica.tar" -C "$INFERNO_ASSETS/mica/"
fi

# InsightFace models
INSIGHTFACE_DIR="$REPO_DIR/models/insightface"
if [ -d "$INSIGHTFACE_DIR/antelopev2" ]; then
    echo "  [SKIP] InsightFace antelopev2 already exists"
else
    echo "  [Download] InsightFace antelopev2..."
    mkdir -p "$INSIGHTFACE_DIR"
    wget -q --show-progress -O "/tmp/antelopev2.zip" \
        "https://keeper.mpdl.mpg.de/f/2d58b7fed5a74cb5be83/?dl=1"
    unzip -q "/tmp/antelopev2.zip" -d "$INSIGHTFACE_DIR/"
    rm -f "/tmp/antelopev2.zip"
fi
echo ""

# ---- 4. MediaPipe HandLandmarker ----
echo "=== MediaPipe Hand Detection ==="
download_if_missing \
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" \
    "$REPO_DIR/models/mediapipe/hand_landmarker.task"
echo ""

# ---- 5. L2CS-Net (Gaze) ----
echo "=== L2CS-Net Gaze Weights ==="
L2CS_WEIGHTS="$REPO_DIR/models/L2CSNet_gaze360.pkl"

if [ -f "$L2CS_WEIGHTS" ]; then
    echo "  [SKIP] Already exists: $L2CS_WEIGHTS"
else
    echo "  L2CS-Net weights must be downloaded manually from Google Drive:"
    echo "    https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd"
    echo "  Download L2CSNet_gaze360.pkl and place it at:"
    echo "    $L2CS_WEIGHTS"
    echo ""
    echo "  Attempting download with gdown..."
    if command -v gdown &>/dev/null; then
        gdown --folder "https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd" -O "$REPO_DIR/models/" 2>/dev/null || \
            echo "  [WARN] gdown failed - please download manually"
    else
        echo "  [WARN] gdown not installed - please download manually"
    fi
fi
echo ""

# ---- 5. SMPL-X / MANO / FLAME (manual registration required) ----
echo "=== Body Models (Manual Registration Required) ==="
echo ""
echo "  The following models require registration at Max Planck Institute:"
echo ""

SMPLX_MODEL="$REPO_DIR/models/smplx/SMPLX_NEUTRAL.npz"
if [ -f "$SMPLX_MODEL" ]; then
    echo "  [OK] SMPL-X:  $REPO_DIR/models/smplx/"
else
    echo "  [ ] SMPL-X:  Register at https://smpl-x.is.tue.mpg.de/"
    echo "               Download SMPL-X v1.1 -> extract to models/smplx/"
fi

MANO_MODEL="$REPO_DIR/models/mano/MANO_RIGHT.pkl"
if [ -f "$MANO_MODEL" ]; then
    echo "  [OK] MANO:    $REPO_DIR/models/mano/"
else
    echo "  [ ] MANO:    Register at https://mano.is.tue.mpg.de/"
    echo "               Download MANO v1.2 -> extract to models/mano/"
fi

FLAME_MODEL="$REPO_DIR/models/flame/generic_model.pkl"
if [ -f "$FLAME_MODEL" ] || [ -d "$REPO_DIR/models/flame" ]; then
    echo "  [OK] FLAME:   $REPO_DIR/models/flame/"
else
    echo "  [ ] FLAME:   Register at https://flame.is.tue.mpg.de/"
    echo "               Download FLAME 2020 -> extract to models/flame/"
    echo "               OR: wget https://download.is.tue.mpg.de/emoca/assets/FLAME.zip"
fi

echo ""
echo "============================================"
echo "  Model download complete!"
echo "============================================"
echo ""
echo "  Check above for any [WARN] or [ ] items that need manual action."
echo ""
