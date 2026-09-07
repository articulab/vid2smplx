# vid2smplx

**Extract full-body SMPL-X parameters from monocular video: body, hands, face, gaze, and blink in a single pipeline.**

<p align="center">
  <img src="assets/hero.gif" alt="vid2smplx output vs original video" width="720"/>
</p>

## Why vid2smplx?

Most video-to-3D methods either recover body pose alone (no hands, no face) or output a single mesh with no way to separately control fingers, jaw, or gaze. [SMPLest-X](https://github.com/sangho-vision/SMPLest-X) is a recent single-model approach that regresses SMPL-X directly, but produces a monolithic mesh without disentangled hand or face articulation.

vid2smplx runs five specialized models in sequence and merges their outputs into a single SMPL-X parameter file with separate, editable channels for body, hands, face, gaze, and blink.

## Pipeline

```
Video --> GVHMR --> HaMeR --> EMICA --> L2CS-Net --> MediaPipe --> Merge --> smplx_params.npz
           body     hands     face      gaze         blink
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
| `left_hand_pose` | (T, 45) | HaMeR, 15 hand joints in axis-angle |
| `right_hand_pose` | (T, 45) | HaMeR, 15 hand joints in axis-angle |
| `jaw_pose` | (T, 3) | EMICA, jaw rotation |
| `expression` | (T, 100) | EMICA, FLAME 2020 expression coefficients |
| `leye_pose` / `reye_pose` | (T, 3) | EMICA, eye rotations |
| `gaze_pitch` / `gaze_yaw` | (T,) | L2CS-Net, gaze direction in radians |
| `blink_left` / `blink_right` | (T,) | MediaPipe, Eye Aspect Ratio (low = closed) |
| `left_hand_valid` / `right_hand_valid` | (T,) | per-frame hand detection mask |
| `face_valid` | (T,) | per-frame face detection mask |
| `gaze_valid` | (T,) | per-frame gaze detection mask |
| `K_fullimg` | (T, 3, 3) | GVHMR, camera intrinsic matrix |
| `num_frames` | scalar | total frame count |
| `coord_system` | string | `"global"` (world-space) |

All outputs are at 30 FPS.

## Quickstart

```bash
git clone https://github.com/articulab/vid2smplx.git && cd vid2smplx   # not --recursive: see docs/install.md
bash install.sh                 # conda env + deps + ~12 GB of weights, ends with `vid2smplx doctor`  (or: --uv)
# doctor will ask for SMPL-X and MANO (free registration) -> see docs/models.md
conda activate vid2smplx
vid2smplx run examples/clip_talking.mp4 --final-incam
```

Result: `output/clip_talking/smplx_params.npz` + `render/final_incam.mp4`.

Needs Linux, an NVIDIA GPU with 8 GB+ VRAM (5.1 GB measured, see [benchmarks](docs/benchmarks.md)), conda or uv, ~15 GB disk. Blackwell GPUs: see [docs/install.md](docs/install.md#blackwell-gpus-rtx-50xx-rtx-pro-sm_120).

## Commands

```
vid2smplx run <video.mp4> [--final-incam] [--full-debug] [--no-face] [--no-hands] [--percent N] [--cleanup]
vid2smplx render <output/clip> --layers final,global,hands,face
vid2smplx doctor                 # env, weights, symlinks -> [OK]/[MISS] table
vid2smplx download               # fetch weights, create symlinks (idempotent)
```

Every step caches, so rerunning a clip resumes where it stopped. Full option list: [docs/usage.md](docs/usage.md).

## Docs

- [Install](docs/install.md) — requirements, conda or uv, Blackwell, troubleshooting, tests
- [Usage](docs/usage.md) — all flags, output tree, loading the npz in Python
- [Models](docs/models.md) — every weight file, where it goes, which need registration
- [Pipeline](docs/pipeline.md) — what each step does and which script runs it
- [Benchmarks](docs/benchmarks.md) — per-stage time, VRAM and GPU utilization; what hardware you need
- [Comparison with SMPLest-X](docs/comparison.md) — GIFs and timings

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

