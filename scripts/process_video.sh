#!/bin/bash
# process_video.sh — End-to-end: video -> SMPL-X params + rendered video
# Usage: bash scripts/process_video.sh <video.mp4> [--production] [--face_method emica] [...]

set -e

# region Noise suppression
export PYTHONWARNINGS="ignore::DeprecationWarning,ignore::FutureWarning"
_filter_noise() {
    grep -v -E "overwriting variable|pkg_resources is deprecated|apex is not installed|Fail to import.*MultiScale|not available in reconstructed resnet|copy resnet state dict|deprecated pixel format|Processing MICA image|UserWarning: To copy construct|UserWarning: You are using a MANO|UserWarning: torch\.(meshgrid|cross|utils\._pytree)|Lipreading model not found|No module named .spectre.|Plan failed with a cudnnException|state keys that would end up colliding|Lightning automatically upgraded|Found keys that are not in the model state|Importing from timm|UserWarning: The parameter .pretrained.|Arguments other than a weight enum|WARN:.*loadsave.*Unsupported depth|unexpected key in source state_dict|do not match exactly|Use load_from_local|UserWarning: torch.cuda.amp|sourceTensor.clone|l2cs.*FutureWarning|MediaPipe.*WARNING" || true
}

# endregion
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Renderer: "nvdr" (nvdiffrast, default — faster) or "pt3d" (pytorch3d)
export RENDERER="${RENDERER:-nvdr}"

CONDA_ENV="${CONDA_ENV:-vid2smplx}"

VIDEO="$1"
if [ -z "$VIDEO" ]; then
    echo "vid2smplx - Video -> SMPL-X Parameters"
    echo ""
    echo "Usage: bash scripts/process_video.sh <video_path> [options]"
    echo ""
    echo "Options:"
    echo "  --percent N            Process only the first N% of the video (for testing)"
    echo "  --downsample N         Take every Nth frame for hand estimation (default: 1)"
    echo "  --dynamic_cam          Use visual odometry (default: static cam)"
    echo "  --batch_size N         HaMeR batch size (default: 48)"
    echo "  --final_incam          Render the final combined incam video (body+hands+face on video)"
    echo "  --full_debug           Render all debug MP4s (incam, global, hands, face)"
    echo "  --output_dir DIR       Custom output directory"
    echo "  --no_hands             Skip hand estimation entirely"
    echo "  --no_face              Skip face tracking (EMICA), gaze, and blink"
    echo "  --cleanup              Delete intermediates after success (keep only smplx_params.npz + gaze_blink)"
    exit 1
fi

shift  # consume video path, rest are options

# region Parse options
DOWNSAMPLE=1
BATCH_SIZE=48
STATIC_CAM="-s"
SKIP_RENDER=0
OUTPUT_BASE=""
PERCENT=100
HAND_DETECTOR="mediapipe"
FACE_METHOD="emica"
NO_HANDS=0
NO_FACE=0
PRODUCTION=1
CLEANUP=0
FINAL_INCAM=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --percent)        PERCENT="$2"; shift 2 ;;
        --downsample)     DOWNSAMPLE="$2"; shift 2 ;;
        --batch_size)     BATCH_SIZE="$2"; shift 2 ;;
        --dynamic_cam)    STATIC_CAM=""; shift ;;
        --skip_render)    SKIP_RENDER=1; shift ;;
        --output_dir)     OUTPUT_BASE="$2"; shift 2 ;;
        --hand_detector)  HAND_DETECTOR="$2"; shift 2 ;;
        --face_method)    FACE_METHOD="$2"; shift 2 ;;
        --no_face)        NO_FACE=1; FACE_METHOD=""; shift ;;
        --use_gvhmr_focal) shift ;; # kept for backwards compat, always on
        --production)     PRODUCTION=1; shift ;; # kept for backwards compat, now default
        --full_debug)     PRODUCTION=0; shift ;;
        --no_hands)       NO_HANDS=1; shift ;;
        --cleanup)        CLEANUP=1; shift ;;
        --final_incam)    FINAL_INCAM=1; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# HAND_DETECTOR: "mediapipe" (fast, default) or "vitpose" (accurate, 3.6GB model)

# endregion

VIDEO="$(realpath "$VIDEO")"
VIDEO_NAME="$(basename "${VIDEO%.*}")"

if [ -z "$OUTPUT_BASE" ]; then
    OUTPUT_BASE="$REPO_DIR/output"
fi
OUTPUT_BASE="$(cd "$REPO_DIR" && mkdir -p "$OUTPUT_BASE" && cd "$OUTPUT_BASE" && pwd)"

# region Trim video if --percent < 100
if [ "$PERCENT" -lt 100 ]; then
    DURATION=$(conda run -n "$CONDA_ENV" --no-capture-output \
        ffprobe -v error -show_entries format=duration \
        -of csv=p=0 "$VIDEO" 2>/dev/null | head -1)
    TARGET_DURATION=$(python3 -c "print(f'{float($DURATION) * $PERCENT / 100:.3f}')")
    TRIM_DIR="$OUTPUT_BASE/.trimmed"
    mkdir -p "$TRIM_DIR"
    TRIMMED_VIDEO="$TRIM_DIR/${VIDEO_NAME}_${PERCENT}pct.mp4"
    if [ ! -f "$TRIMMED_VIDEO" ]; then
        echo "[Trim] Cutting first ${PERCENT}% (${TARGET_DURATION}s of ${DURATION}s)..."
        conda run -n "$CONDA_ENV" --no-capture-output \
            ffmpeg -y -i "$VIDEO" -t "$TARGET_DURATION" -c copy "$TRIMMED_VIDEO" \
            -loglevel warning
    else
        echo "[Trim] Using cached: $TRIMMED_VIDEO"
    fi
    VIDEO="$TRIMMED_VIDEO"
    VIDEO_NAME="${VIDEO_NAME}_${PERCENT}pct"
fi
# endregion

# region Downscale to 1080p if needed
MAX_RES=1080
VIDEO_HEIGHT=$(conda run -n "$CONDA_ENV" --no-capture-output \
    ffprobe -v error -select_streams v:0 \
    -show_entries stream=height -of csv=p=0 "$VIDEO" 2>/dev/null | head -1)
VIDEO_WIDTH=$(conda run -n "$CONDA_ENV" --no-capture-output \
    ffprobe -v error -select_streams v:0 \
    -show_entries stream=width -of csv=p=0 "$VIDEO" 2>/dev/null | head -1)

if [ -n "$VIDEO_HEIGHT" ] && [ -n "$VIDEO_WIDTH" ]; then
    # Check if either dimension exceeds 1920x1080
    LONGER_SIDE=$VIDEO_WIDTH
    [ "$VIDEO_HEIGHT" -gt "$VIDEO_WIDTH" ] && LONGER_SIDE=$VIDEO_HEIGHT

    if [ "$LONGER_SIDE" -gt 1920 ]; then
        DOWNSCALE_DIR="$OUTPUT_BASE/.downscaled"
        mkdir -p "$DOWNSCALE_DIR"
        DOWNSCALED_VIDEO="$DOWNSCALE_DIR/${VIDEO_NAME}_1080p.mp4"
        if [ ! -f "$DOWNSCALED_VIDEO" ]; then
            echo "[Downscale] ${VIDEO_WIDTH}x${VIDEO_HEIGHT} -> 1080p (was $(( VIDEO_WIDTH * VIDEO_HEIGHT / 1000 ))K pixels)..."
            conda run -n "$CONDA_ENV" --no-capture-output \
                ffmpeg -y -i "$VIDEO" \
                -vf "scale='if(gt(iw,ih),1920,-2)':'if(gt(iw,ih),-2,1920)'" \
                -c:v libx264 -crf 18 -preset fast -pix_fmt yuv420p \
                "$DOWNSCALED_VIDEO" -loglevel warning
            echo "[Downscale] Done: $DOWNSCALED_VIDEO"
        else
            echo "[Downscale] Using cached 1080p: $DOWNSCALED_VIDEO"
        fi
        VIDEO="$DOWNSCALED_VIDEO"
    else
        echo "[Resolution] ${VIDEO_WIDTH}x${VIDEO_HEIGHT} — OK (≤1080p)"
    fi
fi
# endregion

OUTPUT_DIR="$OUTPUT_BASE/$VIDEO_NAME"

GVHMR_DIR="$REPO_DIR/GVHMR"
HAMER_DIR="$REPO_DIR/hamer"
SMPLX_DIR="$REPO_DIR/models/smplx"

GVHMR_OUT="$OUTPUT_DIR/gvhmr"
HAMER_OUT="$OUTPUT_DIR/hamer"
HAMER_VIDEO_OUT="$HAMER_OUT/$VIDEO_NAME"
RENDER_OUT="$OUTPUT_DIR/render"
SMPLX_OUT="$OUTPUT_DIR/smplx_params.npz"

mkdir -p "$OUTPUT_DIR"

exec 1> >(_filter_noise) 2> >(_filter_noise >&2)

echo "============================================"
echo "  vid2smplx - Video -> SMPL-X Pipeline"
echo "============================================"
echo "Video:      $VIDEO"
echo "Name:       $VIDEO_NAME"
echo "Output:     $OUTPUT_DIR"
[ "$NO_HANDS" -eq 0 ] && echo "Hand:       HaMeR (detector: ${HAND_DETECTOR})"
[ "$NO_HANDS" -eq 1 ] && echo "Hand:       DISABLED (--no_hands)"
[ -n "$FACE_METHOD" ] && echo "Face:       ${FACE_METHOD}"
echo "Downsample: ${DOWNSAMPLE}x"
echo "Camera:     $([ -n "$STATIC_CAM" ] && echo 'static' || echo 'dynamic')"
[ "$PRODUCTION" -eq 1 ]      && echo "Mode:       PRODUCTION (no renders)"
[ "$FINAL_INCAM" -eq 1 ]     && echo "Render:     final incam only (skip intermediate renders)"
echo "Focal:      use GVHMR intrinsics for hand method"
[ "$PERCENT" -lt 100 ] && echo "Percent:    ${PERCENT}%"
echo ""

# region Timing helper
_timer_start() { eval "_T_$1=$SECONDS"; }
_timer_end() {
    local name="$1"
    local start_var="_T_$name"
    local elapsed=$(( SECONDS - ${!start_var} ))
    echo "  [TIMER] $name: ${elapsed}s"
    eval "TIMER_$name=$elapsed"
}
# endregion

# ---- Step 1: GVHMR (body) ----
echo "==== Step 1/5: GVHMR body estimation ===="
_timer_start gvhmr
GVHMR_RESULT="$GVHMR_OUT/$VIDEO_NAME/hmr4d_results.pt"

if [ -f "$GVHMR_RESULT" ]; then
    echo "  [SKIP] Already exists: $GVHMR_RESULT"
else
    GVHMR_NO_RENDER=""
    ([ "$PRODUCTION" -eq 1 ] || [ "$FINAL_INCAM" -eq 1 ]) && GVHMR_NO_RENDER="--no_render"

    conda run -n "$CONDA_ENV" --no-capture-output --cwd "$GVHMR_DIR" \
        python tools/demo/demo.py \
            --video "$VIDEO" \
            --output_root "$GVHMR_OUT" \
            $STATIC_CAM $GVHMR_NO_RENDER

    if [ ! -f "$GVHMR_RESULT" ]; then
        echo "  [ERROR] GVHMR failed - no output at $GVHMR_RESULT"
        exit 1
    fi
fi
echo "  [OK] Body params: $GVHMR_RESULT"
_timer_end gvhmr

GVHMR_FOCAL=$(conda run -n "$CONDA_ENV" --no-capture-output python -c "
import torch
pred = torch.load('$GVHMR_RESULT', map_location='cpu', weights_only=False)
K = pred['K_fullimg'][0]
print(float(K[0][0]))
")
echo "  [Focal] GVHMR focal length: $GVHMR_FOCAL"
FOCAL_ARG="--focal-length $GVHMR_FOCAL"
echo ""

# ---- Step 2: Hand estimation ----
HAMER_PARAMS="$HAMER_VIDEO_OUT/rendered/hamer_hands.pt"
# Fallback to legacy mano_params/ if .pt doesn't exist
if [ ! -f "$HAMER_PARAMS" ] && [ -d "$HAMER_VIDEO_OUT/rendered/mano_params" ]; then
    HAMER_PARAMS="$HAMER_VIDEO_OUT/rendered/mano_params"
fi

if [ "$NO_HANDS" -eq 1 ]; then
    echo "==== Step 2/5: Hand estimation (SKIPPED --no_hands) ===="
    echo ""
else
    echo "==== Step 2/5: Hand estimation (HaMeR + $HAND_DETECTOR) ===="
    _timer_start hamer

    if [ -f "$HAMER_PARAMS" ] || ([ -d "$HAMER_PARAMS" ] && [ "$(ls -A "$HAMER_PARAMS" 2>/dev/null)" ]); then
        echo "  [SKIP] Already exists: $HAMER_PARAMS"
    else
        # Always skip HaMeR's built-in renders — pipeline does its own hand rendering
        HAMER_NO_RENDER="--no-render"

        GVHMR_BBOXES_ARG=""
        GVHMR_BBX_FILE="$GVHMR_OUT/$VIDEO_NAME/preprocess/bbx.pt"
        if [ -f "$GVHMR_BBX_FILE" ]; then
            GVHMR_BBOXES_ARG="--gvhmr-bboxes $GVHMR_BBX_FILE"
            echo "  [Reuse] Using GVHMR YOLO bboxes (skipping ViTDet): $GVHMR_BBX_FILE"
        fi

        conda run -n "$CONDA_ENV" --no-capture-output \
            python "$SCRIPT_DIR/run_hamer_video.py" \
                --video "$VIDEO" \
                --out_folder "$HAMER_OUT" \
                --corpus-dir "$REPO_DIR" \
                --downsample "$DOWNSAMPLE" \
                --batch-size "$BATCH_SIZE" \
                --hand-detector "$HAND_DETECTOR" \
                $FOCAL_ARG $HAMER_NO_RENDER $GVHMR_BBOXES_ARG

        if [ ! -f "$HAMER_PARAMS" ] && [ ! -d "$HAMER_PARAMS" ]; then
            echo "  [WARN] No hand params saved (no hands detected?)"
        fi
    fi
    echo "  [OK] Hand params: $HAMER_PARAMS"

    if [ "$PRODUCTION" -eq 0 ] && [ "$FINAL_INCAM" -eq 0 ]; then
        HAMER_RENDERED="$HAMER_VIDEO_OUT/rendered"
        if [ -d "$HAMER_RENDERED" ] && [ "$(ls "$HAMER_RENDERED"/*_0.png 2>/dev/null | head -1)" ]; then
            for SUFFIX in 0 1; do
                HAND_VIDEO="$HAMER_RENDERED/hand_${SUFFIX}.mp4"
                if [ -f "$HAND_VIDEO" ]; then
                    echo "  [SKIP] Hand video already exists: $HAND_VIDEO"
                elif [ "$(ls "$HAMER_RENDERED"/*_${SUFFIX}.png 2>/dev/null | wc -l)" -gt 0 ]; then
                    echo "  [ffmpeg] Stitching hand_${SUFFIX}.mp4..."
                    conda run -n "$CONDA_ENV" --no-capture-output \
                        ffmpeg -y -framerate 30 \
                        -pattern_type glob -i "$HAMER_RENDERED/*_${SUFFIX}.png" \
                        -c:v libx264 -pix_fmt yuv420p -crf 18 \
                        "$HAND_VIDEO" -loglevel warning
                fi
            done
        fi

        HANDS_INCAM="$RENDER_OUT/hands_incam.mp4"
        GVHMR_VIDEO="$GVHMR_OUT/$VIDEO_NAME/0_input_video.mp4"
        [ ! -f "$GVHMR_VIDEO" ] && GVHMR_VIDEO="$VIDEO"
        if [ -f "$HAMER_PARAMS" ] || ([ -d "$HAMER_PARAMS" ] && [ "$(ls -A "$HAMER_PARAMS" 2>/dev/null)" ]); then
            if [ ! -f "$HANDS_INCAM" ]; then
                echo "  [Render] Hands incam (pytorch3d)..."
                conda run -n "$CONDA_ENV" --no-capture-output --cwd "$GVHMR_DIR" \
                    python "$SCRIPT_DIR/render_hands_incam.py" \
                        --mano_params "$HAMER_PARAMS" \
                        --video "$GVHMR_VIDEO" \
                        --output "$HANDS_INCAM"
            fi
        fi
    else
        echo "  [Production] Skipping hand renders"
    fi
    _timer_end hamer
    echo ""
fi

# ---- Step 3: FLAME face tracking (EMICA) ----
FLAME_RESULT=""
if [ "$FACE_METHOD" = "emica" ]; then
    echo "==== Step 3/5: FLAME face tracking (EMICA) ===="
    _timer_start emica
    EMICA_OUT="$OUTPUT_DIR/emica"
    FLAME_RESULT="$EMICA_OUT/$VIDEO_NAME/flame_params.npz"

    if [ -f "$FLAME_RESULT" ]; then
        echo "  [SKIP] Already exists: $FLAME_RESULT"
    else
        conda run -n "$CONDA_ENV" --no-capture-output \
            python "$SCRIPT_DIR/run_emica_fast.py" \
                --video "$VIDEO" \
                --out_folder "$EMICA_OUT"

        if [ ! -f "$FLAME_RESULT" ]; then
            echo "  [WARN] EMICA failed - no output, continuing without face"
            FLAME_RESULT=""
        fi
    fi
    if [ -n "$FLAME_RESULT" ]; then
        echo "  [OK] FLAME params: $FLAME_RESULT"
    fi

    if [ "$PRODUCTION" -eq 0 ] && [ "$FINAL_INCAM" -eq 0 ] && [ -n "$FLAME_RESULT" ]; then
        FACE_INCAM="$RENDER_OUT/face_incam.mp4"
        GVHMR_VIDEO="$GVHMR_OUT/$VIDEO_NAME/0_input_video.mp4"
        [ ! -f "$GVHMR_VIDEO" ] && GVHMR_VIDEO="$VIDEO"
        if [ ! -f "$FACE_INCAM" ]; then
            echo "  [Render] Face incam (pytorch3d)..."
            conda run -n "$CONDA_ENV" --no-capture-output --cwd "$GVHMR_DIR" \
                python "$SCRIPT_DIR/render_face_incam.py" \
                    --emica_result "$FLAME_RESULT" \
                    --video "$GVHMR_VIDEO" \
                    --gvhmr_result "$GVHMR_RESULT" \
                    --output "$FACE_INCAM"
        fi
    fi
    _timer_end emica
    echo ""
else
    echo "==== Step 3/5: FLAME face tracking (SKIPPED) ===="
    echo "  [SKIP] No face method specified (use --face_method emica)"
    echo ""
fi

# ---- Step 3.5: Gaze + Blink ----
GAZE_BLINK_OUT="$OUTPUT_DIR/gaze_blink"
GAZE_BLINK_RESULT="$GAZE_BLINK_OUT/$VIDEO_NAME/gaze_blink.npz"

if [ "$FACE_METHOD" = "emica" ]; then
    echo "==== Step 3.5: Gaze + Blink estimation ===="
    _timer_start gaze

    EMICA_CACHE="$OUTPUT_DIR/emica/$VIDEO_NAME/_detection_cache.npz"
    EMICA_CACHE_ARG=""
    [ -f "$EMICA_CACHE" ] && EMICA_CACHE_ARG="--emica_cache $EMICA_CACHE"

    GVHMR_VIDEO="$GVHMR_OUT/$VIDEO_NAME/0_input_video.mp4"
    [ ! -f "$GVHMR_VIDEO" ] && GVHMR_VIDEO="$VIDEO"

    if [ -f "$GAZE_BLINK_RESULT" ]; then
        echo "  [SKIP] Already exists: $GAZE_BLINK_RESULT"
    else
        conda run -n "$CONDA_ENV" --no-capture-output \
            python "$SCRIPT_DIR/run_gaze_blink.py" \
                --video "$GVHMR_VIDEO" \
                --out_folder "$GAZE_BLINK_OUT" \
                --corpus_dir "$REPO_DIR" \
                --video_name "$VIDEO_NAME" \
                $EMICA_CACHE_ARG
    fi
    echo "  [OK] Gaze+Blink: $GAZE_BLINK_RESULT"
    _timer_end gaze
    echo ""
else
    echo "==== Step 3.5: Gaze + Blink (SKIPPED - requires --face_method emica) ===="
    GAZE_BLINK_RESULT=""
    echo ""
fi

# ---- Step 4: Merge -> SMPL-X ----
echo "==== Step 4/5: Merge body + hands + face -> SMPL-X ===="
_timer_start merge

FLAME_ARG=""
[ -n "$FLAME_RESULT" ] && [ -f "$FLAME_RESULT" ] && FLAME_ARG="--flame_result $FLAME_RESULT"

if [ -f "$SMPLX_OUT" ]; then
    echo "  [SKIP] Already exists: $SMPLX_OUT"
else
    HAMER_ARG=""
    if [ -f "$HAMER_PARAMS" ] || ([ -d "$HAMER_PARAMS" ] && [ "$(ls -A "$HAMER_PARAMS" 2>/dev/null)" ]); then
        HAMER_ARG="--hamer_result $HAMER_PARAMS"
    fi

    GAZE_BLINK_ARG=""
    if [ -n "$GAZE_BLINK_RESULT" ] && [ -f "$GAZE_BLINK_RESULT" ]; then
        GAZE_BLINK_ARG="--gaze_blink_result $GAZE_BLINK_RESULT"
    fi

    conda run -n "$CONDA_ENV" --no-capture-output \
        python "$SCRIPT_DIR/merge_body_hands.py" \
            --gvhmr_result "$GVHMR_RESULT" \
            $HAMER_ARG \
            $FLAME_ARG \
            $GAZE_BLINK_ARG \
            --output "$SMPLX_OUT" \
            --coord global
fi
echo "  [OK] SMPL-X params: $SMPLX_OUT"
_timer_end merge
echo ""

# ---- Step 4.5: IK hands (production only) ----
if [ -f "$HAMER_PARAMS" ] || ([ -d "$HAMER_PARAMS" ] && [ "$(ls -A "$HAMER_PARAMS" 2>/dev/null)" ]); then
    if grep -q "ik_wrist_loss" <(python3 -c "import numpy as np; print(' '.join(np.load('$SMPLX_OUT', allow_pickle=True).keys()))" 2>/dev/null); then
        echo "==== Step 4.5: IK hands (SKIPPED — already applied) ===="
    else
        echo "==== Step 4.5: IK hands → match SMPL-X arms to MANO targets ===="
        _timer_start ik
        set +e
        conda run -n "$CONDA_ENV" --no-capture-output --cwd "$GVHMR_DIR" \
            python "$SCRIPT_DIR/ik_hands.py" \
                --smplx_params "$SMPLX_OUT" \
                --hamer_params "$HAMER_PARAMS" \
                --gvhmr_result "$GVHMR_RESULT" \
                --smplx_dir "$SMPLX_DIR"
        IK_EXIT=$?
        set -e
        if [ $IK_EXIT -eq 77 ]; then
            echo "  [FAIL] IK coverage too low — clip quality insufficient"
            echo "IK_COVERAGE_LOW" > "$OUTPUT_DIR/FAILED"
            rm -f "$OUTPUT_DIR/SUCCESS"
            # Send notification if ntfy available
            curl -s -o /dev/null -H "Title: v2s IK FAIL" -H "Priority: high" -H "Tags: x" \
                -d "$VIDEO_NAME: IK coverage below threshold" \
                "https://ntfy.sh/corpus-ymachta" 2>/dev/null || true
            exit 1
        elif [ $IK_EXIT -ne 0 ]; then
            echo "  [ERROR] IK failed with exit code $IK_EXIT"
        else
            echo "  [OK] IK applied to $SMPLX_OUT"
        fi
        _timer_end ik
    fi
    echo ""
fi

# ---- Step 5: Render ----
_timer_start render
if ([ "$SKIP_RENDER" -eq 1 ] || [ "$PRODUCTION" -eq 1 ]) && [ "$FINAL_INCAM" -eq 0 ]; then
    echo "==== Step 5/5: Render (SKIPPED - production/skip_render) ===="
else
    echo "==== Step 5/5: Render body+hands video ===="

    GVHMR_VIDEO="$GVHMR_OUT/$VIDEO_NAME/0_input_video.mp4"
    [ ! -f "$GVHMR_VIDEO" ] && GVHMR_VIDEO="$VIDEO"

    HAMER_RENDER_ARG=""
    if [ -f "$HAMER_PARAMS" ] || ([ -d "$HAMER_PARAMS" ] && [ "$(ls -A "$HAMER_PARAMS" 2>/dev/null)" ]); then
        HAMER_RENDER_ARG="--hamer_params $HAMER_PARAMS"
    fi

    GAZE_RENDER_ARG=""
    if [ -n "$GAZE_BLINK_RESULT" ] && [ -f "$GAZE_BLINK_RESULT" ]; then
        GAZE_RENDER_ARG="--gaze_result $GAZE_BLINK_RESULT"
    fi

    RENDER_FLAGS=""
    [ "$FINAL_INCAM" -eq 1 ] && RENDER_FLAGS="--final_only --no_global"

    conda run -n "$CONDA_ENV" --no-capture-output --cwd "$GVHMR_DIR" \
        python "$SCRIPT_DIR/render_body_hands.py" \
            --gvhmr_result "$GVHMR_RESULT" \
            $HAMER_RENDER_ARG \
            --video "$GVHMR_VIDEO" \
            --output_dir "$RENDER_OUT" \
            --smplx_dir "$SMPLX_DIR" \
            $GAZE_RENDER_ARG $RENDER_FLAGS
    echo "  [OK] Rendered: $RENDER_OUT"
fi
_timer_end render

echo "SUCCESS" > "$OUTPUT_DIR/SUCCESS"

# ---- Cleanup ----
if [ "$CLEANUP" -eq 1 ]; then
    echo "  [Cleanup] Removing intermediates..."
    rm -rf "$GVHMR_OUT" 2>/dev/null
    rm -rf "$HAMER_OUT" 2>/dev/null
    rm -rf "$OUTPUT_DIR/emica" 2>/dev/null
    rm -rf "$RENDER_OUT" 2>/dev/null
    echo "  [Cleanup] Done - kept smplx_params.npz + gaze_blink/"
fi

echo ""
echo "============================================"
echo "  Done! Output:"
echo "============================================"
echo "  SMPL-X params: $SMPLX_OUT"
[ -f "$GAZE_BLINK_RESULT" ] && echo "  Gaze+Blink:   $GAZE_BLINK_RESULT"
if [ "$CLEANUP" -eq 0 ]; then
    [ -d "$RENDER_OUT" ] && echo "  Renders:      $RENDER_OUT/"
fi
echo ""
echo "============================================"
echo "  Step Timings:"
echo "============================================"
[ -n "${TIMER_gvhmr:-}" ]  && echo "  GVHMR:      ${TIMER_gvhmr}s"
[ -n "${TIMER_hamer:-}" ]  && echo "  HaMeR:      ${TIMER_hamer}s"
[ -n "${TIMER_emica:-}" ]  && echo "  EMICA:      ${TIMER_emica}s"
[ -n "${TIMER_gaze:-}" ]   && echo "  Gaze+Blink: ${TIMER_gaze}s"
[ -n "${TIMER_merge:-}" ]  && echo "  Merge:      ${TIMER_merge}s"
[ -n "${TIMER_ik:-}" ]     && echo "  IK Hands:   ${TIMER_ik}s"
[ -n "${TIMER_render:-}" ] && echo "  Render:     ${TIMER_render}s"
echo "  Total:      ${SECONDS}s"
echo ""
