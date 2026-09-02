# Usage

```
vid2smplx run <video.mp4> [options]     video -> smplx_params.npz (+ renders)
vid2smplx render <clip_dir> [options]   re-render layers of an existing output dir
vid2smplx doctor                        check env, weights, symlinks
vid2smplx download                      fetch weights, create symlinks
```

Every step caches its output; rerunning a clip skips finished steps.
`run` executes `doctor` (files only) first and stops early if a weight is missing.

## `vid2smplx run`

| Option | Default | Description |
|--------|---------|-------------|
| `--final-incam` | off | render the final SMPL-X mesh (body+hands+face) over the video |
| `--full-debug` | off | render every intermediate video (body, hands, face, global views) |
| `--no-hands` | off | skip HaMeR |
| `--no-face` | off | skip EMICA, gaze and blink |
| `--percent N` | 100 | process only the first N% (quick tests) |
| `--downsample N` | 1 | run hands on every Nth frame |
| `--batch-size N` | 48 | HaMeR batch size (lower if OOM) |
| `--seed N` | off | fix RNG seeds in the face/gaze/merge/IK steps (GVHMR and HaMeR are deterministic in eval mode); used by the functional tests |
| `--dynamic-cam` | off | moving camera: run visual odometry (default assumes a static camera) |
| `--output-dir DIR` | `output/` | output root |
| `--cleanup` | off | delete intermediates (`gvhmr/`, `hamer/`, `emica/`); keeps `smplx_params.npz`, `gaze_blink/` and `render/` |

Old underscore spellings (`--final_incam`, …) still work. The retired `.sh` flags `--production`, `--skip_render`, `--use_gvhmr_focal` are accepted as no-ops with a deprecation note. `--full-debug` implies `--final-incam`. Bad values (`--percent 0`, missing video) are rejected before anything runs.

Examples:

```bash
vid2smplx run examples/clip_talking.mp4                       # npz only, no renders
vid2smplx run examples/clip_dancing.mp4 --final-incam         # + overlay video
vid2smplx run examples/clip_signing.mp4 --percent 10          # quick test
vid2smplx run my.mp4 --no-face --cleanup --output-dir /data/out
```

## `vid2smplx render`

Re-render from an existing `output/<clip>/` (e.g. after a `--cleanup`-free run):

```bash
vid2smplx render output/clip_talking --layers final,global,hands,face
```

## Output structure

```
output/<video_name>/
├── smplx_params.npz              # merged body+hands+face+gaze+blink  <- the deliverable
├── SUCCESS                       # marker file (FAILED on IK coverage failure)
├── gvhmr/<video_name>/           # hmr4d_results.pt, 0_input_video.mp4
├── hamer/<video_name>/           # per-detection hand params
├── emica/<video_name>/           # flame_params.npz, _detection_cache.npz
├── gaze_blink/<video_name>/      # gaze_blink.npz
└── render/                       # only with --final-incam / --full-debug
    ├── final_incam.mp4           #   full SMPL-X mesh on video
    ├── front.mp4, left.mp4, right.mp4   # global views (--full-debug)
    ├── hands_incam.mp4           #   HaMeR hands only (--full-debug)
    └── face_incam.mp4            #   EMICA face only (--full-debug)
```

## `smplx_params.npz`

All arrays are at 30 FPS. `T` = frame count.

| Key | Shape | Source |
|-----|-------|--------|
| `body_pose` | (T, 63) | GVHMR, 21 body joints, axis-angle |
| `global_orient` | (T, 3) | GVHMR, root orientation |
| `transl` | (T, 3) | GVHMR, global translation |
| `betas` | (T, 10) | GVHMR, body shape |
| `left_hand_pose` / `right_hand_pose` | (T, 45) | HaMeR, 15 joints each, axis-angle (IK-corrected wrists) |
| `jaw_pose` | (T, 3) | EMICA |
| `expression` | (T, 100) | EMICA, FLAME 2020 expression coefficients |
| `leye_pose` / `reye_pose` | (T, 3) | EMICA |
| `gaze_pitch` / `gaze_yaw` | (T,) | L2CS-Net, radians |
| `blink_left` / `blink_right` | (T,) | MediaPipe eye aspect ratio (low = closed) |
| `left_hand_valid` / `right_hand_valid` / `face_valid` / `gaze_valid` | (T,) bool | per-frame detection masks |
| `K_fullimg` | (T, 3, 3) | camera intrinsics |
| `num_frames` | scalar | T |
| `coord_system` | str | `"global"` |

```python
import numpy as np
d = np.load("output/clip_talking/smplx_params.npz", allow_pickle=True)
body = d["body_pose"]                       # (T, 63)
hands = d["left_hand_pose"], d["right_hand_pose"]
gaze = np.stack([d["gaze_pitch"], d["gaze_yaw"]], -1)   # (T, 2)
ok = d["face_valid"] & d["gaze_valid"]      # frames with a detected face
```

Feed the keys straight into `smplx.create(..., model_type="smplx", use_pca=False)`.
