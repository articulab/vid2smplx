# Onboarding audit — 2026-09-06

How one-shottable is this repo? Tested by deleting the cleps tree and rebuilding both
install paths from an empty clone. **Neither could build a working environment.** Both do
now. This page records what was wrong and how each was diagnosed, so the reasoning does
not have to live in the source.

Result: `bash install.sh --force` on cleps → install/doctor/unit/CLI all exit 0, seven
stages `[OK]`, ik_coverage 97.4%. `install_blackwell.sh --force` locally → verifier passes,
`doctor` all-`[OK]`.

## Why it was broken without anyone noticing

Both working environments were accreted by hand over months. The installers were never
what actually built them, so they drifted freely. `install_blackwell.sh` could not have
worked as committed — `REPO_DIR` pointed at `specific_installation/`, so every
`pip install -e "$REPO_DIR/GVHMR"` targeted a path that does not exist.

## Dependency defects (cleps path)

Each surfaced only after the previous was fixed: Python resolves imports eagerly, and
inferno's `FaceRecBase` transitively imports emotion, audio, logging and ONNX-conversion
modules the pipeline never uses.

| package | symptom |
|---|---|
| mmpose | `KeyError: 'ViT is not in the models registry'` — must be the **ViTPose fork** in `third-party/`, not PyPI mmpose. Both call themselves 0.24.0; only the fork registers the ViT backbone. |
| mediapipe | 1.0 removed `mediapipe.solutions`; unpinned installs broke blink EAR + EMICA crop |
| protobuf | 5+ dropped `FieldDescriptor.label`, which mediapipe's `solution_base` reads → `AttributeError` naming neither library |
| h5py | inferno reaches it through `hickle` (installed `--no-deps`) |
| json_tricks, munkres, terminaltables | mmpose/mmdet module-level imports |
| imgaug, shapely, sk-video | inferno transforms and video IO |
| wandb | `FaceRecBase` imports it unconditionally; pinned 0.25.x because >=0.26 needs protobuf>=5 while mediapipe needs <4 |
| soundfile, librosa, loguru | `lightning_logging` / `EmoCnnModule`, reached via the EmoNetLoss chain — the *face* model cannot load without an *audio* library |
| onnx2torch | MICA's RetinaFace |

**Ordering matters more than the pins.** The safety-net `numpy`/`protobuf` pin has to run
*after* the editable installs. Pinning it in Phase 3 was silently undone by Phase 4
re-resolving dependencies — observed in one run: protobuf 3.20.3 → 7.36.1 between phases.

**The two paths need different pin sets.** torch 2.3/numpy 1.23 (cleps) vs torch 2.10/
numpy 1.26 (Blackwell). Imposing one set on both broke insightface — first an ABI error
(`numpy.dtype size changed`), then the opposite complaint (`no attribute 'exceptions'`).

## Structural defects

- **`doctor` gave a false `[OK]` while HaMeR was dead.** `hamer_demo_data.tar.gz` ships an
  *empty* `_DATA/data/mano/`. The symlink step skipped it and the check passed it, both
  because the path existed. Same class as the earlier `MANO_SMPLX_vertex_ids.pkl` bug that
  silently disabled IK. Presence checks now require the target to be non-empty.
- **`git clone --recursive` cannot work for anyone** — inferno declares nested submodules
  on a private GitLab, so recursion aborts the clone. `install.sh` inits submodules correctly.
- **Blackwell source builds silently produced CPU-only binaries.** No `nvcc` in the env, so
  pytorch3d built without CUDA and failed much later as `RuntimeError: Not compiled with GPU
  support` from the renderer. Fixed by installing conda `cuda-toolkit` and asserting CUDA in
  the verifier.
- **A verifier piped through `grep` loses its exit code.** `set -e` reads the last command
  of a pipeline, so `sys.exit(1)` was discarded and a broken env printed "installation
  complete!".
- **`install_blackwell.sh` contradicted `requirements_blackwell.txt`** on 12 packages. Note
  the frozen file is itself stale (dated 2026-03-23, older than the env's torch), so it is a
  consistency reference, not ground truth. `tests/test_cli.py` now enforces they agree.
- Source builds default to one `nvcc` job per core (~2-3 GB each) and OOM-kill a 30 GB
  laptop. Capped via `MAX_JOBS`.

## Reproducibility

Measured on the 1.5 s functional clip, seed 0:

| varied | `body_pose` max abs diff |
|---|---|
| GPU only (V100 vs RTX 6000, same torch) | 0.0063 rad |
| GPU + torch (cleps 2.3 vs Blackwell 2.9) | 0.1645 rad |

Cross-stack joint displacement on a byte-identical clip: **5.3 mm mean, 16.5 mm max** —
inside GVHMR's own ~50-80 mm benchmark error, so both are valid, but systematic.
**Process one corpus on one stack.** The 96-clip interpersonality corpus is all cleps.

Not a seeding issue: `scripts/utils.py` already sets `cudnn.deterministic`. The residual is
float non-associativity — different architectures and cuDNN versions pick different
reduction orders — amplified by network depth and by `ik_hands.py` being a 150-iteration
optimiser (`transl` differs 0.006 while `body_pose`, which IK rewrites, differs 0.16).
TF32 is never disabled and is a plausible additional contributor on Ampere+; unmeasured.

## Why hands vary across GPUs, and why the golden compares p99 not max

Hand poses differ across GPU architectures by far more than body poses.

**What is measured.** Same clip and seed, V100 vs RTX 6000:

| key | frames | agree to 1e-6 | >0.01 rad | max |
|---|---|---|---|---|
| `right_hand_pose` | 38 | 32 | 1 | 0.0795 |
| `body_pose` | 38 | 0 | 0 | 0.0063 |

Both drift on every frame; hands additionally show a long tail in axis-angle terms.

**Traced end to end. Verified facts, in order:**

1. *The pipeline is exactly deterministic on fixed hardware.* Same V100, same allocation, same
   clip, run twice back to back: **76/76 detections bitwise identical, 0.000 mm**. Also
   confirms the `--seed` help text's claim that GVHMR/HaMeR are deterministic in eval.
   So none of this is run-to-run nondeterminism.
2. *Inputs are identical across the two GPU runs* — same clip file (md5), same env, same code;
   only the GPU differs (gpu001 V100 vs gpu002 RTX 6000).
3. *The detector does produce discrete flips, from FP16.* `vitpose_detect.py:68` runs the
   heatmap under `autocast(float16)`. FP16's 10-bit mantissa makes exact ties between adjacent
   cells common, and `flat.max(dim=2)` then resolves the tie by reduction order. Measured: one
   keypoint moved **exactly 30.00 px** (= 1920/64, one heatmap cell, portrait 1080x1920) with
   its confidence **exactly equal** at 0.822265625; 1 of 80 bboxes shifted 30 px.
4. *But those flips did not cause the pose differences.* The one detection whose bbox changed
   had `pose_maxdiff = 0.00000`, and the largest pose difference (frame 2, 5.88 mm) had an
   identical bbox and identical keypoints.
5. *So the residual arises inside HaMeR's fp32 forward pass on an identical crop*: different
   architectures use different cuDNN kernels and reduction orders. Most detections stay at
   ~1e-6 (57/76); a few amplify. **Which operation amplifies has not been traced.**

Note GVHMR guards precision explicitly (`@autocast(enabled=False)` on rotary embeddings and
in the pipeline) while the hand detector opts into FP16 — the body path has a guard the hand
path lacks.

**The large axis-angle numbers are a representation artifact.** That worst detection moves the
hand mesh by mean 1.89 mm / max 5.88 mm — physically negligible — while reading 0.066 in
rotation-matrix terms and ~0.1 rad in axis-angle. Axis-angle is ill-conditioned for the small
rotations typical of finger joints: the axis direction is poorly defined as the angle goes to
zero, so a tiny rotation change produces a large coordinate change. Hands sit in that regime;
the body root and limbs do not.

So the golden should not compare axis-angle `max` at all — it is a noisy statistic over an
ill-conditioned representation. The 99th percentile is used because it is robust to the tail
that representation produces, while still failing when every frame shifts:

| | `right_hand_pose` max | p99 |
|---|---|---|
| GPU noise (V100 vs RTX 6000) | 0.0795 — fails | **0.0124 — passes** |
| Real regression (pre-fix golden) | 0.2997 — fails | **0.2456 — fails** |

p99 separates them by 20x where max separated them by 3.8x and failed both. Verified: the uv
env on an RTX 8000 passes 9/9 against a golden generated on a V100.

**Latent quality issue, separate from the above:** the detector takes the raw argmax with no
sub-pixel refinement (no soft-argmax / DARK decoding), so hand keypoints are quantised to the
heatmap grid. This did NOT cause the cross-GPU differences measured here, but it is a real
accuracy limitation worth fixing on its own; it changes hand output for every clip and needs
its own change and goldens.

## Test-harness flaws found while measuring the above

- The functional fixture **re-encoded** the clip with libx264, so the input video differed
  per machine — 1,077,323 bytes locally vs 1,047,970 on cleps. Every cross-machine golden
  comparison therefore had an uncontrolled input difference. Now `-c copy`.
- `tests/functional/golden/smplx_params.npz` was committed 2026-08-31 in `07723ac`, which
  **predates** `38d73fa` (the pipeline correctness fixes: `ik_hands.py` rewrite, left-hand
  MANO mirror, gaze ROI). `test_golden` was failing because behaviour deliberately changed
  and the reference was never re-accepted. The diffs map onto the fixes one-to-one:
  `left_hand_pose` ↔ mirror fix, `body_pose` ↔ IK rewrite, `blink`/`gaze` ↔ ROI fix.
