# Model weights

`vid2smplx doctor` prints this same list with an `[OK]` / `[MISS]` next to each file.
`vid2smplx download` fetches every **auto** row and creates the symlinks below.

| Group | File (relative to repo) | How to get it |
|-------|-------------------------|---------------|
| GVHMR | `GVHMR/inputs/checkpoints/{gvhmr,hmr2,vitpose,yolo,dpvo}/*` | auto |
| HaMeR | `hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt`, `hamer/_DATA/data/mano_mean_params.npz` | auto |
| EMICA | `models/inferno/FaceReconstruction/`, `models/inferno/mica/model/mica.tar` | auto |
| EMICA | `~/.insightface/models/antelopev2/` | auto (inferno hard-codes this location) |
| EMICA | `inferno/assets/FLAME/geometry/generic_model.pkl` | auto (`FLAME.zip` from EMOCA, unzipped into `inferno/assets/`) |
| Hands | `models/mediapipe/hand_landmarker.task` | auto |
| Gaze | `models/L2CSNet_gaze360.pkl` | `gdown` tries; else download manually from [Google Drive](https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd) into `models/` |
| SMPL-X | `models/smplx/SMPLX_NEUTRAL.npz` | **manual** — register at https://smpl-x.is.tue.mpg.de/, download *SMPL-X v1.1 (NPZ)*, unzip into `models/smplx/` |
| MANO | `models/mano/MANO_RIGHT.pkl` (+ `MANO_LEFT.pkl`) | **manual** — register at https://mano.is.tue.mpg.de/, download *MANO v1.2*, unzip into `models/mano/` |

Registration is free for research use. ~15 GB total.

## Symlinks

The submodules hard-code their own asset locations, so `vid2smplx download` links them to `models/`:

```
GVHMR/inputs/checkpoints/body_models/smplx  ->  models/smplx
hamer/_DATA/data/mano                       ->  models/mano
inferno/assets/FaceReconstruction           ->  models/inferno/FaceReconstruction
inferno/assets/MICA                         ->  models/inferno/mica
```

If `doctor` shows a `link:` row as `[MISS]`, rerun `vid2smplx download` (it is idempotent and skips finished downloads).

## Final layout

```
models/
├── smplx/SMPLX_NEUTRAL.npz          (manual)
├── mano/MANO_RIGHT.pkl              (manual)
├── inferno/{FaceReconstruction,mica}/
├── mediapipe/hand_landmarker.task
└── L2CSNet_gaze360.pkl
```
