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
| `--cleanup` | off | delete intermediates (`gvhmr/`, `hamer/`, `emica/`); keeps `smplx_params.npz`, `gaze_blink/` and `render/` |

**Reuse and re-runs.** Each stage writes a `.stamp.json` recording the input file (size + mtime) and the flags that shaped it. A stage is reused only when both still match, so replacing a video with a different one under the same filename recomputes instead of returning the old result. `[STALE] Recomputing <stage>: <reason>` says why whenever that happens.

**Markers.** `SUCCESS`, `FAILED` and `summary.json` are cleared at the start of every run and exactly one marker is left at the end — including after a crash or Ctrl-C, where `FAILED` names the stage and the reason. A directory with no marker means the run was never attempted.

**More than one person.** GVHMR reconstructs a single body, and hands/face/gaze run their own detectors that do not share that choice. A video with a **second subject** therefore fails with `FAILED` naming every track it saw. Crop to one person, or pass `--person N` to accept a specific one — the caveat about hands/face/gaze still applies and is recorded in `summary.json`.

A YOLO track counts as a second subject only when it is **co-present with the chosen track for >= 5% of frames** *and* its **median bbox covers >= 1% of the frame** (`MULTI_PERSON_OVERLAP_FRAC` / `MULTI_PERSON_MIN_AREA` in `vid2smplx/cli.py`). Below that it is a blip — a reflection, a passer-by, or a YOLO id switch that re-identified the same person — and the run proceeds on the dominant track with a loud `warnings` entry naming the track and its extent. Co-presence, not duration, is the discriminator: an id switch produces a long track over *disjoint* frames, and must never fail a batch job. The area gate is deliberately absolute, not relative to the chosen track: a real second subject sitting further from the camera measures ~0.24x the chosen bbox and still fails.

Old underscore spellings (`--final_incam`, …) still work. The retired `.sh` flags `--production`, `--skip_render`, `--use_gvhmr_focal` are accepted as no-ops with a deprecation note. `--full-debug` implies `--final-incam`. Bad values (`--percent 0`, missing video) are rejected before anything runs.

Examples:

```bash
vid2smplx run examples/clip_talking.mp4                       # npz only, no renders
vid2smplx run examples/clip_dancing.mp4 --final-incam         # + overlay video
vid2smplx run examples/clip_signing.mp4 --percent 10          # quick test
vid2smplx run my.mp4 --no-face --cleanup --output-dir /data/out
```

## Interrupted runs: what survives a kill

**Re-running the exact same command is always the right move after a crash, a `scancel`,
a SLURM preemption or a time limit.** It reuses every finished stage and redoes only the
one that was interrupted. You never need to delete the output directory first.

How that is guaranteed:

- Every stage writes its artifact to a temp file in the destination directory and
  `os.replace`s it into place on success. `os.replace` is atomic, so the final path only
  ever holds the *old* complete file or the *new* complete one — never a half-written one.
  This matters because `scancel` and preemption arrive as SIGKILL, which cannot be caught.
- Each artifact is stamped (`<name>.stamp.json`) with the input identity and the flags that
  shaped it, and the stamp is written *after* the artifact is complete. A stage killed
  mid-way therefore has no stamp, and is redone.
- A stamped `.npz` is additionally opened before it is trusted. A stamp proves provenance,
  not that the bytes are whole — this catches artifacts left behind by older versions.

### What cannot be resumed

Resumption is **per stage**, not within a stage. If a stage is killed at 95 %, that stage
starts over from the beginning.

The one that hurts is **GVHMR**, the longest stage (over an hour on a 20-min video):

- Its preprocessing sub-steps *are* checkpointed and reused — tracking (`bbx.pt`), 2-D pose
  (`vitpose.pt`), ViT features (`vit_features.pt`) and SLAM (`slam.pt`). A kill after these
  have been written does not repeat them.
- Its final **HMR4D pass is not checkpointed**. It runs over the whole sequence in one shot,
  and a kill during it means that pass restarts from zero. There is no partial-credit
  mechanism, and the pipeline does not pretend there is one.

So the worst case is: everything before HMR4D is reused, HMR4D is redone in full — and that
is cheap. Measured on a 20-min, 35,755-frame video on an rtx8000: the whole GVHMR stage took
1 h 18 m, of which the HMR4D pass was **27.5 s**. The time is all in the checkpointed
preprocessing (tracking 26 min, ViTPose 29 min, ViT features 15 min).

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

Arrays keep the source video's frame rate (no stage resamples); it is stored in the `fps` key. `T` = frame count.

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
| `gaze_pitch` / `gaze_yaw` | (T,) | L2CS-Net, radians. Zero unless the run used `--gaze` |
| `blink_left` / `blink_right` | (T,) | MediaPipe eye aspect ratio (low = closed). Zero unless the run used `--gaze` |
| `left_hand_valid` / `right_hand_valid` / `face_valid` / `gaze_valid` | (T,) bool | per-frame detection masks |
| `K_fullimg` | (T, 3, 3) | camera intrinsics |
| `num_frames` | scalar | T |
| `coord_system` | str | `"global"` |
| `fps` | scalar | source frame rate (0 if ffprobe could not read it) |

```python
import numpy as np
d = np.load("output/clip_talking/smplx_params.npz", allow_pickle=True)
body = d["body_pose"]                       # (T, 63)
hands = d["left_hand_pose"], d["right_hand_pose"]
gaze = np.stack([d["gaze_pitch"], d["gaze_yaw"]], -1)   # (T, 2)
ok = d["face_valid"] & d["gaze_valid"]      # frames with a detected face
```

Feed the keys straight into `smplx.create(..., model_type="smplx", use_pca=False)`.
