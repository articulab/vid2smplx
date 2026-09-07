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
| Gaze | `models/L2CSNet_gaze360.pkl` | auto (mirror; the upstream Drive folder 404s) |
| SMPL-X | `models/smplx/SMPLX_NEUTRAL.npz`, `models/smplx/MANO_SMPLX_vertex_ids.pkl` | **manual** — register at https://smpl-x.is.tue.mpg.de/, download *SMPL-X v1.1 (NPZ)*. Both files are in the archive; the IK stage fails without the vertex ids |
| MANO | `models/mano/MANO_RIGHT.pkl` (+ `MANO_LEFT.pkl`) | **manual** — register at https://mano.is.tue.mpg.de/, download *MANO v1.2*, unzip into `models/mano/` |

Registration is free for research use. ~15 GB total.

The gaze weight is mirrored because the upstream Google Drive folder now 404s:
`https://huggingface.co/ymachta/articumotion-checkpoints/resolve/main/mirrors/L2CSNet_gaze360.pkl`
(95,849,977 bytes, md5 `a0fb3d74cab1ab1a4435876be5483321`; `vid2smplx download` fetches
and checksums it automatically).

`models/` starts empty (git ignores its subdirectories), so create the folders first:
`mkdir -p models/smplx models/mano`. The SMPL-X archive expands to a nested
`models/smplx/` of its own — copy the files in, don't unzip the archive on top, or you
end up with `models/smplx/models/smplx/` and `doctor` still reports `[MISS]`.

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
