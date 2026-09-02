#!/bin/bash
#SBATCH --job-name=interp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# VRAM, not host RAM, is the binding constraint. GVHMR's HMR4D transformer runs over
# the WHOLE sequence: a 35k-frame video tries to allocate ~38GB on the GPU. Measured
# on this dataset: rtx8000 (46GB, shared 3-way) OOM, v100 (32GB) OOM, **a100 (40GB) OOM**
# (tasks 2,3 on gpu013). Only h100 (80GB) and h200 (141GB) succeed. Do not relax
# without shortening the sequence. A 1-min clip fits on 8GB — need scales with length.
#SBATCH --constraint="h100|h200"
#SBATCH --exclude=gpu018
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-95%1
#SBATCH --output=/scratch/ymachta/interpersonality_out/logs/%A_%a.out
#SBATCH --error=/scratch/ymachta/interpersonality_out/logs/%A_%a.out
#
# One 20-min interpersonality video per array task, strictly one at a time (%1).
#   video -> SMPL-X params -> 1-min highest-motion excerpt render
# Staged inputs are never modified; only derived data under $OUT is written.

set -u
ROOT=/scratch/ymachta
STAGE=$ROOT/interpersonality_staging
OUT=$ROOT/interpersonality_out
SCRIPTS=$ROOT/Corpus/vid2smplx/scripts
PY=$ROOT/miniconda3/envs/vid2smplx/bin/python
# render_excerpt.py shells out to ffmpeg, which lives in the env, not on the node PATH.
# Without this the excerpt render dies with FileNotFoundError: 'ffmpeg'.
export PATH="$ROOT/miniconda3/envs/vid2smplx/bin:$PATH"
MIN_FREE_GB=200          # refuse to start if scratch is this tight

mkdir -p "$OUT/logs"

# Deterministic video list (sorted, so task N always means the same file)
# LIST is overridable so short/long clips can be submitted as separate arrays with
# different --constraint: VRAM need scales with sequence length, so 12.8k-frame FT
# clips fit on a100/rtx8000 while 36k-frame BP/CS/TP need h100/h200.
LIST=${LIST:-$OUT/videos.txt}
if [ ! -s "$LIST" ]; then
    find "$STAGE" -name '*.mp4' | sort > "$LIST"
fi
VIDEO=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST")
[ -n "$VIDEO" ] || { echo "[skip] no video for task $SLURM_ARRAY_TASK_ID"; exit 0; }

NAME=$(basename "${VIDEO%.*}")
CLIP=$OUT/$NAME

echo "=== [$SLURM_ARRAY_TASK_ID] $NAME on $(hostname) $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

# Idempotent: already finished -> skip
if [ -f "$CLIP/DONE" ]; then
    echo "[skip] already done: $CLIP"
    exit 0
fi

# Disk guard — a full /scratch mid-run corrupts nothing here, but wastes GPU hours
FREE_GB=$(df -BG --output=avail /scratch/ymachta | tail -1 | tr -dc '0-9')
if [ "$FREE_GB" -lt "$MIN_FREE_GB" ]; then
    echo "[ABORT] only ${FREE_GB}G free on /scratch (< ${MIN_FREE_GB}G)"
    exit 1
fi
echo "[disk] ${FREE_GB}G free"

mkdir -p "$CLIP"

# ---- 1. pipeline: video -> smplx_params.npz --------------------------------
echo "--- process_video ---"
# strip tqdm carriage-return spam: 35k-frame bars produced 660KB logs per task
$PY "$SCRIPTS/process_video.py" "$VIDEO" --output_dir "$CLIP" --cleanup \
    2>&1 | tr '\r' '\n' | grep -vE "it/s\]|it/s,|s/it\]"
rc=${PIPESTATUS[0]}

# process_video.py nests its output as <output_dir>/<video_name>/
NPZ="$CLIP/$NAME/smplx_params.npz"
[ -f "$NPZ" ] || NPZ="$CLIP/smplx_params.npz"

if [ $rc -ne 0 ] || [ ! -f "$NPZ" ]; then
    echo "[FAIL] process_video rc=$rc, no smplx_params.npz"
    echo "process_video rc=$rc" > "$CLIP/FAILED"
    exit 1
fi

# Hands are the point of this corpus — a params file without them is not success.
HANDS=$($PY -c "
import numpy as np,sys
z=np.load('$NPZ',allow_pickle=True)
v=z['left_hand_valid'].mean()+z['right_hand_valid'].mean() if 'left_hand_valid' in z else 0
print(f'{v/2:.3f}')" 2>/dev/null || echo 0)
echo "[check] hand coverage: $HANDS"
if [ "$(echo "$HANDS < 0.05" | bc -l 2>/dev/null)" = "1" ]; then
    echo "[FAIL] hand coverage $HANDS — HaMeR produced nothing"
    echo "no_hands coverage=$HANDS" > "$CLIP/FAILED"
    exit 1
fi

# ---- 2. 1-min highest-motion excerpt ---------------------------------------
echo "--- excerpt render ---"
$PY "$SCRIPTS/render_excerpt.py" \
    --npz "$NPZ" \
    --video "$VIDEO" \
    --out "$CLIP/${NAME}_excerpt.mp4" \
    --seconds 60 \
  || echo "[WARN] excerpt render failed (params are still valid)"

echo "OK $(date)" > "$CLIP/DONE"
echo "=== done $NAME: $(du -sh "$CLIP" | cut -f1) ==="
