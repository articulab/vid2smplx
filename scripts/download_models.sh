#!/bin/bash
# ============================================================================
# download_models.sh - Download all model weights for vid2smplx
#
# Usage:
#   vid2smplx download   (or: bash scripts/download_models.sh)
#   vid2smplx download   (or: bash scripts/download_models.sh)
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
GVHMR_DIR="$REPO_DIR/GVHMR"
HAMER_DIR="$REPO_DIR/hamer"
# Staging dir for archives: NOT /tmp — that is often a small tmpfs and the archives are multi-GB
TMP="$REPO_DIR/models/.downloads"
mkdir -p "$TMP"

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
    wget -q --show-progress -O "$TMP/hamer_demo_data.tar.gz" \
        "https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz"
    tar -xzf "$TMP/hamer_demo_data.tar.gz" -C "$HAMER_DIR"   # archive already contains the _DATA/ top level
    rm -f "$TMP/hamer_demo_data.tar.gz"
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
    wget -q --show-progress -O "$TMP/FaceReconstruction.zip" \
        "https://download.is.tue.mpg.de/emote/FaceReconstruction.zip"
    unzip -q "$TMP/FaceReconstruction.zip" -d "$INFERNO_ASSETS/"
    rm -f "$TMP/FaceReconstruction.zip"
fi

# MICA model (a PyTorch checkpoint; despite the .tar name it is not an archive)
MICA_CKPT="$INFERNO_ASSETS/mica/model/mica.tar"
if [ -f "$MICA_CKPT" ]; then
    echo "  [SKIP] MICA model already exists"
else
    echo "  [Download] MICA model..."
    mkdir -p "$(dirname "$MICA_CKPT")"
    wget -q --show-progress -O "$MICA_CKPT" \
        "https://keeper.mpdl.mpg.de/f/db172dc4bd4f4c0f96de/?dl=1"
fi

# FLAME assets for EMICA (EMOCA release: head template, masks, generic_model.pkl) -> inferno/assets/FLAME
if [ -f "$REPO_DIR/inferno/assets/FLAME/geometry/generic_model.pkl" ]; then
    echo "  [SKIP] inferno/assets/FLAME already exists"
else
    echo "  [Download] FLAME assets (EMOCA)..."
    mkdir -p "$REPO_DIR/inferno/assets"
    wget -q --show-progress -O "$TMP/FLAME.zip" "https://download.is.tue.mpg.de/emoca/assets/FLAME.zip"
    unzip -q -o "$TMP/FLAME.zip" -d "$REPO_DIR/inferno/assets/"
    rm -f "$TMP/FLAME.zip"
fi

# InsightFace antelopev2 — inferno loads it from ~/.insightface (hard-coded), so put it there
INSIGHTFACE_DIR="$HOME/.insightface/models/antelopev2"
if [ -f "$INSIGHTFACE_DIR/scrfd_10g_bnkps.onnx" ]; then
    echo "  [SKIP] InsightFace antelopev2 already exists"
else
    echo "  [Download] InsightFace antelopev2 -> $INSIGHTFACE_DIR"
    mkdir -p "$INSIGHTFACE_DIR"
    wget -q --show-progress -O "$TMP/antelopev2.zip" \
        "https://keeper.mpdl.mpg.de/f/2d58b7fed5a74cb5be83/?dl=1"
    unzip -q -o -j "$TMP/antelopev2.zip" -d "$INSIGHTFACE_DIR/"
    rm -f "$TMP/antelopev2.zip"
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
        gdown --folder "17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd" -O "$REPO_DIR/models/" 2>/dev/null || true
        [ -f "$L2CS_WEIGHTS" ] || echo "  [WARN] gdown could not fetch it (Google Drive rate-limits folder listings) - download manually, see above"
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

echo ""
echo "============================================"
echo "  Model download complete!"
echo "============================================"
echo ""
rmdir "$TMP" 2>/dev/null || true
echo "  Check above for any [WARN] or [ ] items that need manual action."
echo ""
