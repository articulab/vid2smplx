#!/bin/bash
#SBATCH --job-name=redo_ex
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu018
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ymachta/interpersonality_out/logs/redo_%A_%a.out
#SBATCH --error=/scratch/ymachta/interpersonality_out/logs/redo_%A_%a.out
#
# Re-render excerpts that failed with ffmpeg exit 187. Params are already valid;
# this only regenerates the preview video. No GPU-type constraint needed — the
# renderer FKs one frame at a time, so VRAM is small regardless of clip length.

set -u
ROOT=/scratch/ymachta
OUT=$ROOT/interpersonality_out
SCRIPTS=$ROOT/Corpus/vid2smplx/scripts
PY=$ROOT/miniconda3/envs/vid2smplx/bin/python
export PATH="$ROOT/miniconda3/envs/vid2smplx/bin:$PATH"   # ffmpeg lives here

LIST=$OUT/redo_excerpts.txt
NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST")
[ -n "$NAME" ] || { echo "[skip] no entry for task $SLURM_ARRAY_TASK_ID"; exit 0; }

CLIP=$OUT/$NAME
NPZ=$CLIP/$NAME/smplx_params.npz
VIDEO=$(grep -m1 "/${NAME}\.mp4$" $OUT/videos.txt)

echo "=== redo excerpt: $NAME on $(hostname) ==="
[ -f "$NPZ" ]   || { echo "[FAIL] no params at $NPZ"; exit 1; }
[ -f "$VIDEO" ] || { echo "[FAIL] no source video for $NAME"; exit 1; }

rm -rf "$CLIP/_ex"                      # clear any half-written frames from the failed run

$PY "$SCRIPTS/render_excerpt.py" \
    --npz "$NPZ" --video "$VIDEO" \
    --out "$CLIP/${NAME}_excerpt.mp4" --seconds 60
rc=$?

# A zero-byte file is a failure, not a result — don't leave one lying around.
if [ $rc -ne 0 ] || [ ! -s "$CLIP/${NAME}_excerpt.mp4" ]; then
    echo "[FAIL] excerpt still failed for $NAME (rc=$rc)"
    rm -f "$CLIP/${NAME}_excerpt.mp4"
    rm -rf "$CLIP/_ex"
    exit 1
fi
rm -rf "$CLIP/_ex"
echo "[OK] $NAME $(du -h "$CLIP/${NAME}_excerpt.mp4" | cut -f1)"
