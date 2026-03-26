# vid2smplx/scripts

Video-to-SMPL-X parameter extraction pipeline: body, hands, face, gaze, and blink.

## Pipeline

| Step | What | Script |
|------|------|--------|
| 1 | Body pose + global trajectory (GVHMR) | `process_video.py` calls GVHMR internally |
| 2 | Hand pose (HaMeR) | `run_hamer_video.py` |
| 3 | Face + gaze + blink (EMICA + L2CS + MediaPipe) | `run_emica.py`, `run_gaze_blink.py` |
| 4 | Merge + IK hand correction | `merge_body_hands.py`, `ik_hands.py` |
| 5 | Render overlays | `render.py` |

`process_video.py` orchestrates all steps sequentially.

## Usage

Full pipeline:
```bash
python process_video.py video.mp4 --output_dir output/
```

Production (no renders, delete intermediates):
```bash
python process_video.py video.mp4 --output_dir output/ --cleanup
```

Render layers after production:
```bash
python render.py --clip_dir output/clip_name --layers gvhmr,hands,face,final
```

Final incam only:
```bash
python process_video.py video.mp4 --output_dir output/ --final_incam
```

## Output structure

```
output/clip_name/
  smplx_params.npz    — merged SMPL-X (body+hands+face+gaze+body_valid)
  gaze_blink.npz      — per-frame gaze+blink
  gvhmr/              — GVHMR intermediates (deleted with --cleanup)
  hamer/              — HaMeR intermediates (deleted with --cleanup)
  emica/              — EMICA intermediates (deleted with --cleanup)
  render/             — rendered videos
    final_incam.mp4   — body + IK hands + face + gaze
    gvhmr_incam.mp4   — raw GVHMR body (debug)
    hands_incam.mp4   — HaMeR hands only (debug)
    face_incam.mp4    — EMICA face only (debug)
    front/left/right.mp4 — global triview (debug)
```

## smplx_params.npz contents

Body (from GVHMR):
- `body_pose` (N, 63), `global_orient` (N, 3), `betas` (N, 10), `transl` (N, 3)
- `global_orient_incam`, `transl_incam` — incam-coordinate variants (when coord_system=global)
- `body_valid` (N,) bool — False for frames where GVHMR depth estimation failed

Hands (from HaMeR, SLERP-interpolated):
- `left_hand_pose` (N, 45), `right_hand_pose` (N, 45)
- `left_hand_valid` (N,) bool, `right_hand_valid` (N,) bool

Face (from EMICA/FLAME):
- `jaw_pose` (N, 3), `expression` (N, 100)
- `leye_pose` (N, 3), `reye_pose` (N, 3)
- `face_valid` (N,) bool

Gaze + blink:
- `gaze_pitch` (N,), `gaze_yaw` (N,), `gaze_valid` (N,) bool
- `blink_left` (N,), `blink_right` (N,)

IK (if ik_hands.py ran):
- `body_pose_incam` (N, 63), `left_hand_pose_incam` (N, 45), `right_hand_pose_incam` (N, 45)
- `global_orient_incam` (N, 3), `transl_incam` (N, 3), `betas_incam` (N, 10)
- `ik_wrist_loss`, `ik_finger_loss`, `ik_coverage`, `ik_n_targets`

Metadata:
- `K_fullimg` — camera intrinsics
- `num_frames`, `coord_system`

## Scripts

| File | Description |
|------|-------------|
| `process_video.py` | End-to-end orchestrator: runs all steps via subprocess |
| `run_hamer_video.py` | HaMeR hand estimation on video frames |
| `run_emica.py` | EMICA FLAME face parameter extraction |
| `run_gaze_blink.py` | L2CS gaze + MediaPipe blink detection |
| `merge_body_hands.py` | Merge GVHMR + HaMeR + FLAME + gaze into smplx_params.npz |
| `ik_hands.py` | Inverse kinematics to correct SMPL-X wrist/finger pose from HaMeR targets |
| `render.py` | Render overlays (incam body/hands/face/gaze, global triview) |
| `utils.py` | Shared utilities: rotations, smoothing, MANO loading, renderer selection |
| `hand_utils.py` | Hand mesh loading, incam-to-global transform, direct-coupling placement |
