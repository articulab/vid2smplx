# Pipeline internals

```
Video --> GVHMR --> HaMeR --> EMICA --> [L2CS-Net --> MediaPipe] --> Merge --> IK --> smplx_params.npz
           body     hands     face    gaze        blink  (--gaze)
```

| Step | What | Code |
|------|------|------|
| 0 | trim (`--percent`), downscale >1080p | `vid2smplx/cli.py` |
| 1 | body pose + global trajectory | `GVHMR/tools/demo/demo.py` |
| 2 | hand pose (HaMeR, reusing GVHMR person bboxes) | `scripts/run_hamer_video.py` |
| 3 | face (EMICA / FLAME) | `scripts/run_emica.py` |
| 3.5 | gaze (L2CS-Net) + blink (MediaPipe EAR) — opt-in, `--gaze` | `scripts/run_gaze_blink.py` |
| 4 | merge into one SMPL-X npz | `scripts/merge_body_hands.py` |
| 4.5 | wrist IK so HaMeR hands sit on the GVHMR arms | `scripts/ik_hands.py` |
| 5 | renders (`--final-incam` / `--full-debug`) | `scripts/render.py` |

`vid2smplx/cli.py` orchestrates: each step runs as a subprocess, checks for its own output file and
is skipped when that already exists. Inside the env (`conda activate`, or `source .venv/bin/activate`
— the only case under `--uv`) steps run with the current interpreter; from outside any env they wrap
in `conda run -n $CONDA_ENV`. `vid2smplx/checks.py` is the single list of every weight file and
symlink; `doctor`, `download` and `run`'s pre-flight all read it.

Shared helpers live in `scripts/utils.py` (SMPL-X forward, MANO loading, filters) and `scripts/hand_utils.py`.
