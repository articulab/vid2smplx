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

# ---- Helpers ----
# Size floors come from MODELS in vid2smplx/checks.py -- one table, read here, never copied.
MIN_BYTES_TABLE="$(python3 -c "
import ast
mod = ast.parse(open('$REPO_DIR/vid2smplx/checks.py').read())
for n in mod.body:
    if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'MODELS':
        # Per-ELEMENT literal_eval, never the whole tuple: the note field is a
        # module constant (DOWNLOADABLE), i.e. an ast.Name, and literal_eval raises
        # on any non-literal in the tree. Only fields 1 (path) and 3 (min_bytes)
        # are needed here and both are always literals.
        for e in n.value.elts:
            print(ast.literal_eval(e.elts[1]), ast.literal_eval(e.elts[3]))
")"

min_bytes_for() {   # absolute path -> its floor, or 1 (any non-empty file) when untabulated
    local rel="${1#$REPO_DIR/}"
    rel="${rel/#$HOME\//\~/}"
    awk -v k="$rel" '$1==k {print $2; f=1} END{if(!f) print 1}' <<< "$MIN_BYTES_TABLE"
}

actual_bytes() {
    [ -e "$1" ] || { echo 0; return; }
    if [ -d "$1" ]; then du -sb "$1" | cut -f1; else stat -c %s "$1"; fi
}

have_enough() {     # true only when the target exists AND meets its floor
    local size; size=$(actual_bytes "$1")
    [ "$size" -ge "$(min_bytes_for "$1")" ]
}

fetch() {           # url dest -- download via .part so an interrupted wget never lands at dest
    local url="$1" dest="$2"
    mkdir -p "$(dirname "$dest")"
    rm -f "$dest.part"
    wget -q --show-progress -O "$dest.part" "$url"
    mv -f "$dest.part" "$dest"
}

download_if_missing() {
    local url="$1" dest="$2"
    if have_enough "$dest"; then
        echo "  [SKIP] Already exists: $dest"
        return
    fi
    if [ -e "$dest" ]; then
        echo "  [REDO] $dest is $(actual_bytes "$dest") bytes, expected >= $(min_bytes_for "$dest") — truncated, re-fetching"
        rm -rf "$dest"
    fi
    echo "  [Download] $dest"
    fetch "$url" "$dest"
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

if have_enough "$HAMER_DATA/hamer_ckpts/checkpoints/hamer.ckpt"; then
    echo "  [SKIP] Already exists: hamer.ckpt"
else
    echo "  [Download] HaMeR demo data..."
    rm -f "$HAMER_DATA/hamer_ckpts/checkpoints/hamer.ckpt"
    fetch "https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz" \
        "$TMP/hamer_demo_data.tar.gz"
    tar -xzf "$TMP/hamer_demo_data.tar.gz" -C "$HAMER_DIR"   # archive already contains the _DATA/ top level
    rm -f "$TMP/hamer_demo_data.tar.gz"
    # hamer/models/__init__.py::download_models skips its 5.7 GB fetch only when this
    # exact path exists -- and on a miss it runs `tar -xvf` with no -C, extracting into
    # the CWD and CLOBBERING the models/mano symlink. Leave a zero-byte sentinel: it is
    # never read, only stat()ed. Without it every fresh clone silently loses its hands.
    : > "$HAMER_DIR/_DATA/hamer_demo_data.tar.gz"
fi
echo ""

# ---- 3. EMICA / Inferno weights ----
echo "=== EMICA / Inferno Weights ==="
INFERNO_ASSETS="$REPO_DIR/models/inferno"

# FaceReconstruction models
if have_enough "$INFERNO_ASSETS/FaceReconstruction/models"; then
    echo "  [SKIP] FaceReconstruction already exists"
else
    echo "  [Download] FaceReconstruction models..."
    mkdir -p "$INFERNO_ASSETS"
    fetch "https://download.is.tue.mpg.de/emote/FaceReconstruction.zip" "$TMP/FaceReconstruction.zip"
    unzip -q -o "$TMP/FaceReconstruction.zip" -d "$INFERNO_ASSETS/"
    rm -f "$TMP/FaceReconstruction.zip"
fi

# MICA model (a PyTorch checkpoint; despite the .tar name it is not an archive)
MICA_CKPT="$INFERNO_ASSETS/mica/model/mica.tar"
download_if_missing "https://keeper.mpdl.mpg.de/f/db172dc4bd4f4c0f96de/?dl=1" "$MICA_CKPT"

# FLAME assets for EMICA (EMOCA release: head template, masks, generic_model.pkl) -> inferno/assets/FLAME
if have_enough "$REPO_DIR/inferno/assets/FLAME/geometry/generic_model.pkl"; then
    echo "  [SKIP] inferno/assets/FLAME already exists"
else
    echo "  [Download] FLAME assets (EMOCA)..."
    mkdir -p "$REPO_DIR/inferno/assets"
    fetch "https://download.is.tue.mpg.de/emoca/assets/FLAME.zip" "$TMP/FLAME.zip"
    unzip -q -o "$TMP/FLAME.zip" -d "$REPO_DIR/inferno/assets/"
    rm -f "$TMP/FLAME.zip"
fi

# InsightFace antelopev2 — inferno loads it from ~/.insightface (hard-coded), so put it there
INSIGHTFACE_DIR="$HOME/.insightface/models/antelopev2"
if have_enough "$INSIGHTFACE_DIR/scrfd_10g_bnkps.onnx"; then
    echo "  [SKIP] InsightFace antelopev2 already exists"
else
    echo "  [Download] InsightFace antelopev2 -> $INSIGHTFACE_DIR"
    mkdir -p "$INSIGHTFACE_DIR"
    fetch "https://keeper.mpdl.mpg.de/f/2d58b7fed5a74cb5be83/?dl=1" "$TMP/antelopev2.zip"
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
# The upstream Google-Drive folder 404s (link rot) and gdown cannot list folders
# reliably anyway. The weight is mirrored in the ArticuMotion checkpoints repo and
# fetched here exactly like every other HF-hosted weight above.
# Gaze is opt-in at run time (`--gaze`); the weight is 85 MB, so it is still fetched here
# rather than leaving a later --gaze run to fail on a missing file.
echo "=== L2CS-Net Gaze Weights ==="
L2CS_WEIGHTS="$REPO_DIR/models/L2CSNet_gaze360.pkl"
L2CS_SHA256="8a7f3480d868dd48261e1d59f915b0ef0bb33ea12ea00938fb2168f212080665"

download_if_missing \
    "https://huggingface.co/ymachta/articumotion-checkpoints/resolve/main/mirrors/L2CSNet_gaze360.pkl" \
    "$L2CS_WEIGHTS"

if [ -f "$L2CS_WEIGHTS" ] && command -v sha256sum &>/dev/null; then
    got=$(sha256sum "$L2CS_WEIGHTS" | cut -d' ' -f1)
    if [ "$got" != "$L2CS_SHA256" ]; then
        echo "  [WARN] checksum mismatch for $L2CS_WEIGHTS"
        echo "         expected $L2CS_SHA256"
        echo "         got      $got"
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
