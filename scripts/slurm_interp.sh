#!/bin/bash
#SBATCH --job-name=interp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# VRAM is the binding constraint, but it is no longer quadratic in length: GVHMR 8be5155
# bands the HMR4D attention window instead of building a dense (L,L) mask. Measured on this
# dataset: a 35,755-frame video peaks at 13,710 MiB and returns rc=0 on rtx8000 (46GB), where
# it previously died on a 38.10GiB allocation. 13.7GB is flat from 721 to 35,755 frames, so
# ~16GB is enough at any length; 16GB cards are INFERRED, not observed. rtx8000 allocates in
# minutes on cleps while h100 is a multi-day queue, hence the default below.
# cleps-specific defaults. #SBATCH lines are read by sbatch before any shell runs, so they
# cannot use $OUT or any other variable -- override them on the SUBMIT line instead, which
# always wins over a directive here:
#   sbatch --constraint=a100 --exclude="" --output="$OUT/logs/%A_%a.out" scripts/slurm_interp.sh
#SBATCH --constraint="rtx8000|a100|h100|h200"
#SBATCH --exclude=gpu018
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --array=0-95%5
# Preemption/time-limit safety. Every stage artifact is written temp-then-renamed and
# only stamped once complete, so a killed attempt never leaves a file a re-run trusts:
# requeueing simply resumes at the first unfinished stage.
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
# Logs default to the SUBMISSION directory, which every submitter can write. They used to
# be hardcoded to /scratch/ymachta/interpersonality_out/logs/, so anyone else's array died
# before it started -- and slurm gives no output at all when it cannot open the log file.
# To put them beside the results instead: sbatch --output="$OUT/logs/%A_%a.out" ...
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.out
#
# One 20-min interpersonality video per array task, 5 tasks at a time (%5).
#
# Concurrency and walltime are DEFAULTS, not policy -- the submit line always wins over
# the directives above, so override both without editing this file:
#   sbatch --array=0-$((N-1))%8 --time=12:00:00 scripts/slurm_interp.sh
# Why %5: %1 made the corpus pay 96 SEQUENTIAL queue waits. Measured over 11,706 gpu jobs,
# the queue median is 29 min (rtx8000) / 34 min (h100) with a p90 of 15 h / 36 h, so the
# waiting, not the computing, set the corpus wall clock. QOS `normal` caps 8 concurrent
# GPUs; 5 leaves headroom for interactive work.
# Why 6 h: median clip is 1.4 h, and a short walltime is what lets the backfill scheduler
# slot a task into a gap instead of queueing it behind a 24 h reservation. Overrunning is
# not data loss -- --requeue plus temp-then-rename staging resumes at the first unfinished
# stage (see the USR1 handler below).
#   video -> SMPL-X params -> 1-min highest-motion excerpt render
# Staged inputs are never modified; only derived data under $OUT is written.

set -u
ROOT=${ROOT:-/scratch/$USER}
STAGE=${STAGE:-$ROOT/interpersonality_staging}   # input folder  (override to point elsewhere)
OUT=${OUT:-$ROOT/interpersonality_out}           # output folder (override to point elsewhere)
# REPO defaults to this checkout (the dir holding scripts/), so the array uses the
# env you actually built -- not the retired conda env this script used to hardcode.
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
[ -x "$REPO/.venv/bin/vid2smplx" ] || echo "[warn] no venv at $REPO/.venv — set REPO=<checkout>"
SCRIPTS=$REPO/scripts
PY=$REPO/.venv/bin/python
[ -x "$PY" ] || PY=$ROOT/miniconda3/envs/vid2smplx/bin/python   # conda fallback
# ffmpeg/ffprobe live in the env's bin (install.sh vendors them there), not on the
# node PATH. Without this the excerpt render dies with FileNotFoundError: 'ffmpeg'.
export PATH="$(dirname "$PY"):$PATH"
MIN_FREE_GB=200          # refuse to start if scratch is this tight

mkdir -p "$OUT/logs"

# --- preemption / time-limit handler ---------------------------------------
# B:USR1@120 fires in THIS shell ~120s before the limit. bash defers a trap until the
# current foreground command returns, so the pipeline runs in the background and we
# wait on it -- otherwise the handler could not run until the stage it is warning about
# had already finished.
say_resumable() {
    echo "[SIGNAL] Partial artifacts are safe to keep: each stage writes to a temp file"
    echo "[SIGNAL] and renames on success, and is stamped only when complete."
    echo "[SIGNAL] Re-running the SAME command on ${CLIP:-the output dir} reuses every"
    echo "[SIGNAL] finished stage and redoes only the interrupted one. GVHMR's preprocess"
    echo "[SIGNAL] steps (bbx, vitpose, vit_features, slam) are cached too; its final"
    echo "[SIGNAL] HMR4D pass is NOT resumable and restarts from the beginning."
}
on_usr1() {          # time limit / preemption warning -> ask for another attempt
    echo "[SIGNAL] USR1 at $(date): preempted or near the time limit."
    [ -n "${PIPE_PID:-}" ] && kill -TERM "$PIPE_PID" 2>/dev/null
    say_resumable
    scontrol requeue "$SLURM_JOB_ID" 2>/dev/null || echo "[SIGNAL] could not requeue"
    exit 143
}
on_term() {          # scancel: the user meant it -- exit cleanly, never requeue
    echo "[SIGNAL] TERM at $(date): cancelled."
    [ -n "${PIPE_PID:-}" ] && kill -TERM "$PIPE_PID" 2>/dev/null
    say_resumable
    exit 143
}
trap on_usr1 USR1
trap on_term TERM

# Deterministic video list (sorted, so task N always means the same file)
# LIST is overridable so short/long clips can be submitted as separate arrays with
# different --constraint. VRAM no longer scales with length (see the header), so this is
# now only useful for splitting by wall time, not by GPU size.
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
FREE_GB=$(df -BG --output=avail "$OUT" | tail -1 | tr -dc '0-9')
if [ -z "$FREE_GB" ]; then
    echo "[ABORT] could not read free space on $OUT — does it exist and is it mounted?"
    exit 1
fi
if [ "$FREE_GB" -lt "$MIN_FREE_GB" ]; then
    echo "[ABORT] only ${FREE_GB}G free on /scratch (< ${MIN_FREE_GB}G)"
    exit 1
fi
echo "[disk] ${FREE_GB}G free on $OUT"

mkdir -p "$CLIP"

# ---- 1. pipeline: video -> smplx_params.npz --------------------------------
echo "--- process_video ---"
# strip tqdm carriage-return spam: 35k-frame bars produced 660KB logs per task
# backgrounded so the USR1 trap above can actually fire mid-stage; the subshell exits
# with the pipeline's status, not grep's (grep -v exits 1 when it filters everything).
(
    "$(dirname "$PY")/vid2smplx" run "$VIDEO" --output-dir "$CLIP" --cleanup \
        2>&1 | tr '\r' '\n' | grep -vE "it/s\]|it/s,|s/it\]"
    exit ${PIPESTATUS[0]}
) &
PIPE_PID=$!
wait "$PIPE_PID"
rc=$?

# the CLI nests its output as <output_dir>/<video_name>/
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
