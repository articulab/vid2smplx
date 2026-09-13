# vid2smplx

**Extract full-body SMPL-X parameters from monocular video: body, hands and face in a single pipeline — plus experimental, opt-in gaze and blink (`--gaze`).**

<p align="center">
  <img src="assets/hero.gif" alt="vid2smplx output vs original video" width="720"/>
</p>

## Why vid2smplx?

Most video-to-3D methods recover body pose alone, or output a single mesh with no way to control fingers, jaw or gaze separately — including [SMPLest-X](https://github.com/sangho-vision/SMPLest-X), which regresses SMPL-X directly but as a monolithic mesh.

vid2smplx runs specialized models in sequence and merges them into one SMPL-X parameter file with separate, editable channels for body, hands and face, plus gaze and blink behind `--gaze` (experimental — see [Known limitations](#known-limitations)).

## Pipeline

```
Video --> GVHMR --> HaMeR --> EMICA --> [L2CS-Net --> MediaPipe] --> Merge --> IK --> smplx_params.npz
           body     hands     face    gaze        blink  (--gaze)              wrists
```

Each step caches its output. Rerunning skips completed steps.

## Output

A single `.npz` per video:

| Parameter | Shape | Source |
|-----------|-------|--------|
| `body_pose` | (T, 63) | GVHMR, 21 body joints in axis-angle |
| `global_orient` | (T, 3) | GVHMR, root orientation |
| `transl` | (T, 3) | GVHMR, global translation |
| `betas` | (T, 10) | GVHMR, body shape |
| `left_hand_pose` / `right_hand_pose` | (T, 45) | HaMeR, 15 joints each in axis-angle (IK-corrected wrists) |
| `jaw_pose` | (T, 3) | EMICA, jaw rotation |
| `expression` | (T, 100) | EMICA, FLAME 2020 expression coefficients |
| `leye_pose` / `reye_pose` | (T, 3) | EMICA, eye rotations |
| `gaze_pitch` / `gaze_yaw` | (T,) | L2CS-Net, gaze direction in radians. Requires `--gaze` |
| `blink_left` / `blink_right` | (T,) | MediaPipe, Eye Aspect Ratio (low = closed). Requires `--gaze` |
| `left_hand_valid` / `right_hand_valid` | (T,) | per-frame hand detection mask |
| `face_valid` | (T,) | per-frame face detection mask |
| `gaze_valid` | (T,) | per-frame gaze detection mask. All `False` without `--gaze` |
| `K_fullimg` | (T, 3, 3) | GVHMR, camera intrinsic matrix |
| `num_frames` | scalar | total frame count |
| `coord_system` | string | `"global"` (world-space) |
| `fps` | scalar | frame rate of the source video |

No stage resamples: outputs keep the source video's frame rate, which is stored in `fps`. Frame index alone is not a time base — this corpus mixes 25 and 29.97.

## Quickstart

```bash
git clone git@github.com:articulab/vid2smplx.git   # private repo: SSH. NOT --recursive
cd vid2smplx
bash install.sh                 # env + deps + ~16 GB of weights, ends with `vid2smplx doctor`  (or: --uv)
# doctor asks for SMPL-X and MANO (free registration) -> docs/models.md
conda activate vid2smplx
vid2smplx run examples/clip_talking.mp4 --final-incam
```

Result: `output/clip_talking/smplx_params.npz` + `render/final_incam.mp4`. Details:
[docs/install.md](docs/install.md).

Needs Linux, conda or uv, ~23 GB disk, and an NVIDIA GPU: **8 GB** up to 12,526 frames (~8 min at 25 fps), **~16 GB** beyond that — peak follows the card, not the clip ([benchmarks](docs/benchmarks.md)). Blackwell GPUs: [docs/install.md](docs/install.md#blackwell-gpus-rtx-50xx-rtx-pro-sm_120).

## Commands

```
vid2smplx run <video.mp4> [--final-incam] [--full-debug] [--no-face] [--no-hands] [--gaze] [--percent N] [--cleanup]
vid2smplx render <output/clip> --layers final,global,hands,face
vid2smplx doctor                 # env, weights, symlinks, submodule patches -> [OK]/[MISS] table
vid2smplx setup                  # verify GVHMR is at the pinned fork commit (install.sh does this)
vid2smplx download               # fetch weights, create symlinks (idempotent)
```

Every step caches, so rerunning a clip resumes where it stopped. Full option list: [docs/usage.md](docs/usage.md).

## Docs

- [Install](docs/install.md) — conda or uv, Blackwell, troubleshooting, tests
- [Usage](docs/usage.md) — all flags, output tree, loading the npz
- [Models](docs/models.md) — every weight file and which need registration
- [Pipeline](docs/pipeline.md) — each step and the script that runs it
- [Benchmarks](docs/benchmarks.md) — time, VRAM, and what hardware you need
- [Comparison with SMPLest-X](docs/comparison.md) — GIFs and timings

## Known limitations

What these mean for the numbers in your `smplx_params.npz`:

- **Eye pose is a proxy, not a measurement.** `leye_pose` / `reye_pose` are a normalised 2-D
  iris offset scaled by a fixed constant, written into a slot SMPL-X reads as axis-angle
  radians (`scripts/run_emica.py`). Nothing calibrates that scale, so the magnitudes are not
  radians in any meaningful sense. Hence gaze and blink are opt-in (`--gaze`), and the default
  run leaves these channels at zero.
- **Blink validity follows the face box, not the landmarks.** `gaze_blink.npz` marks a frame
  valid when a face bbox exists, even when MediaPipe found no eye landmarks in it
  (`scripts/run_gaze_blink.py`). Those frames carry `blink_left`/`blink_right` of exactly
  `0.0`, indistinguishable from a closed eye. Treat an exact `0.0` as missing, not a blink.
- **One person per video.** The npz holds one body and one left/right hand per frame; with
  several people in frame the extra hand detections overwrite each other
  (`scripts/merge_body_hands.py`) while the debug overlay renders all of them. If the overlay
  shows more hands than one person has, do not trust the npz. `run` refuses a multi-person
  clip unless you pass `--person N`.
- **IK output is not checked for finiteness.** `scripts/ik_hands.py` writes solved poses
  straight to the npz; a diverged solve puts `NaN` in the file. Check
  `np.isfinite(d["body_pose"]).all()` before using a result.

## Models and citations

- **[GVHMR](https://github.com/zju3dv/GVHMR)**: World-grounded human motion recovery. Liang et al., SIGGRAPH Asia 2024.
- **[HaMeR](https://github.com/geopavlakos/hamer)**: Hand mesh recovery. Pavlakos et al., CVPR 2024.
- **[EMICA](https://github.com/radekd91/inferno)**: Emotion-driven face reconstruction via FLAME. Danecek et al., CVPR 2022.
- **[L2CS-Net](https://github.com/Ahmednull/L2CS-Net)**: Gaze estimation. Abdelrahman et al., 2023.
- **[SMPL-X](https://smpl-x.is.tue.mpg.de/)**: Expressive body model. Pavlakos et al., CVPR 2019.
- **[MediaPipe](https://developers.google.com/mediapipe)**: Face mesh for blink detection (Eye Aspect Ratio).

## License

MIT. See [LICENSE](LICENSE).

The body models (SMPL-X, MANO, FLAME) have their own licenses from Max Planck Institute and are for research purposes only.

