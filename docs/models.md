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
| Gaze | `models/L2CSNet_gaze360.pkl` | auto (HuggingFace mirror `ymachta/articumotion-checkpoints`, sha256-checked by `scripts/download_models.sh`) |
| SMPL-X | `models/smplx/SMPLX_NEUTRAL.npz` **and** `models/smplx/MANO_SMPLX_vertex_ids.pkl` | **manual** — register at https://smpl-x.is.tue.mpg.de/, download *SMPL-X v1.1 (NPZ)*, unzip into `models/smplx/`. Both files ship in that one archive; the vertex-ids file is easy to miss if you extract only the NPZ, and `scripts/ik_hands.py` hard-fails without it |
| MANO | `models/mano/MANO_RIGHT.pkl` (+ `MANO_LEFT.pkl`) | **manual** — register at https://mano.is.tue.mpg.de/, download *MANO v1.2*, unzip into `models/mano/` |

Registration is free for research use.

**Size (measured):** ~15 GB in the repo + 1.4 GB in `~/.insightface` = **~16 GB of weights**;
~23 GB for the complete install ([install.md](install.md)).

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
├── smplx/MANO_SMPLX_vertex_ids.pkl  (manual, same archive — the IK stage needs it)
├── mano/MANO_RIGHT.pkl              (manual)
├── inferno/{FaceReconstruction,mica}/
├── mediapipe/hand_landmarker.task
└── L2CSNet_gaze360.pkl
```
