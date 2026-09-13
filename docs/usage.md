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
| `--gaze` | **off** | also estimate gaze and blink. Experimental and uncalibrated — see [Known limitations](../README.md#known-limitations). Ignored with `--no-face`. |
| `--percent N` | 100 | process only the first N% (quick tests) |
| `--downsample N` | 1 | run hands on every Nth frame |
| `--batch-size N` | 48 | HaMeR batch size (lower if OOM) |
| `--seed N` | off | fix RNG seeds in the face/gaze/merge/IK steps (GVHMR and HaMeR are deterministic in eval mode); used by the functional tests |
| `--dynamic-cam` | off | moving camera: run visual odometry (default assumes a static camera) |
| `--person N` | – | which person to reconstruct when several are in frame (rank by size, 0 = largest). A multi-person video is refused without it. |
| `--output-dir DIR` | `output/` | output root |
| `--cleanup` | off | delete intermediates (`gvhmr/`, `hamer/`, `emica/`); keeps `smplx_params.npz`, `render/` and `gaze_blink/` if it exists |

**Reuse and re-runs.** Each stage stamps its artifact (`.stamp.json`) with the input file (size + mtime) and the flags that shaped it, and is reused only when both still match — so a different video under the same filename recomputes. `[STALE] Recomputing <stage>: <reason>` says why.

**Markers.** `SUCCESS`, `FAILED` and `summary.json` are cleared at the start of every run and exactly one marker is left at the end, including after a crash or Ctrl-C (`FAILED` names the stage and the reason). No marker means the run was never attempted.

**More than one person.** GVHMR reconstructs a single body; hands/face/gaze run their own detectors and do not share that choice. A video with a **second subject** fails with `FAILED` naming every track it saw. Crop to one person, or pass `--person N` to accept one — the hands/face/gaze caveat still applies and is recorded in `summary.json`.

A YOLO track counts as a second subject only when it is **co-present with the chosen track for >= 5% of frames** *and* its **median bbox covers >= 1% of the frame** (`MULTI_PERSON_OVERLAP_FRAC` / `MULTI_PERSON_MIN_AREA` in `vid2smplx/cli.py`). Below that it is a blip — a reflection, a passer-by, or a YOLO id switch — and the run proceeds on the dominant track with a loud `warnings` entry. Co-presence, not duration, is the discriminator: an id switch produces a long track over *disjoint* frames and must never fail a batch job. The area gate is absolute, not relative: a real second subject further from the camera measures ~0.24x the chosen bbox and still fails.

Old underscore spellings (`--final_incam`, …) still work. The retired `.sh` flags `--production`, `--skip_render`, `--use_gvhmr_focal` are accepted as no-ops with a deprecation note. `--full-debug` implies `--final-incam`. Bad values (`--percent 0`, missing video) are rejected before anything runs.

Examples:

```bash
vid2smplx run examples/clip_talking.mp4                       # npz only, no renders
vid2smplx run examples/clip_dancing.mp4 --final-incam         # + overlay video
vid2smplx run examples/clip_signing.mp4 --percent 10          # quick test
vid2smplx run my.mp4 --no-face --cleanup --output-dir /data/out
```

## Interrupted runs: what survives a kill

**Re-run the exact same command after a crash, a `scancel`, a preemption or a time limit.** It
reuses every finished stage and redoes only the interrupted one; never delete the output dir first.

Why it is safe: every stage writes to a temp file and `os.replace`s it into place, so the final path
holds either the old complete file or the new one — never a half-written one (SIGKILL cannot be
caught). The stamp is written *after* the artifact, so a stage killed mid-way has no stamp and is
redone, and a stamped `.npz` is opened before it is trusted.

Resumption is **per stage**, not within one: a stage killed at 95 % starts over. GVHMR is the
longest (over an hour on a 20-min video), but its preprocessing *is* checkpointed — tracking
(`bbx.pt`), 2-D pose (`vitpose.pt`), ViT features (`vit_features.pt`), SLAM (`slam.pt`). Only the
final HMR4D pass restarts from zero, and it is cheap: measured on a 20-min, 35,755-frame video on an
rtx8000, the GVHMR stage took 1 h 18 m of which HMR4D was **27.5 s** (tracking 26 min, ViTPose
29 min, ViT features 15 min).

## `vid2smplx render`

Re-render from an existing `output/<clip>/` (e.g. after a `--cleanup`-free run):

```bash
vid2smplx render output/clip_talking --layers final,global,hands,face
```

## Output structure

```
output/<video_name>/
├── smplx_params.npz              # merged body+hands+face (+gaze/blink with --gaze)  <- the deliverable
├── SUCCESS                       # marker file (FAILED on IK coverage failure)
├── gvhmr/<video_name>/           # hmr4d_results.pt, 0_input_video.mp4
├── hamer/<video_name>/           # per-detection hand params
├── emica/<video_name>/           # flame_params.npz, _detection_cache.npz
├── gaze_blink/<video_name>/      # gaze_blink.npz — only with --gaze
└── render/                       # only with --final-incam / --full-debug
    ├── final_incam.mp4           #   full SMPL-X mesh on video
    ├── front.mp4, left.mp4, right.mp4   # global views (--full-debug)
    ├── hands_incam.mp4           #   HaMeR hands only (--full-debug)
    └── face_incam.mp4            #   EMICA face only (--full-debug)
```

## `smplx_params.npz`

Every key, its shape and its source: [README](../README.md#output). Arrays keep the source video's
frame rate (no stage resamples); it is in the `fps` key (0 if ffprobe could not read it). `T` =
frame count. `gaze_pitch`/`gaze_yaw`/`blink_*` are zero and `gaze_valid` all-`False` unless the run
used `--gaze`.

```python
import numpy as np
d = np.load("output/clip_talking/smplx_params.npz", allow_pickle=True)
body = d["body_pose"]                       # (T, 63)
hands = d["left_hand_pose"], d["right_hand_pose"]
gaze = np.stack([d["gaze_pitch"], d["gaze_yaw"]], -1)   # (T, 2)
ok = d["face_valid"] & d["gaze_valid"]      # frames with a detected face
```

Feed the keys straight into `smplx.create(..., model_type="smplx", use_pca=False)`.
