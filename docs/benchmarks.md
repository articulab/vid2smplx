# Benchmarks

Every number below is labelled **MEASURED** (with the card and date) or **INFERRED**.
Peak VRAM is above the idle baseline; utilization is the mean `nvidia-smi` GPU-util during
the stage. Raw numbers: [benchmarks.json](benchmarks.json).

## How much VRAM you actually need

**8 GB is the minimum and ~16 GB is what you want for long video.** Those are not in conflict,
because peak VRAM is set mostly by *the card*, not by the clip: EMICA, HaMeR, ViTPose and the
HMR2 feature pass all size their batches from **free VRAM** (`GVHMR/hmr4d/utils/auto_batch.py`),
so the same clip expands to fill a big card and contracts on a small one.

MEASURED, `scripts/bench.py examples/clip_talking.mp4`, whole-pipeline peak:

| Card | 39 frames | 150 frames | 369 frames |
|---|---:|---:|---:|
| RTX PRO 1000 Laptop, **8 GB** (2026-09, Blackwell env) | — | — | **5.1 GB** |
| Quadro RTX 8000, **46 GB** (2026-09-13, gpu006, GVHMR `ea4ba35`) | **7.3 GB** | **8.3 GB** | **12.3 GB** |

The same 369-frame clip peaks at 5.1 GB on an 8 GB card and 12.3 GB on a 46 GB card, and both
return rc=0. A bigger peak on a bigger card is the heuristic working, not a leak.

**Where clip length does come in.** GVHMR's HMR4D pass is the only stage whose working set grows
with frame count, and since banded attention (GVHMR `8be5155`) it grows slowly. MEASURED
2026-09-13 on the rtx8000 (gpu006, GVHMR `ea4ba35`, `--no-hands --no-face` so the figure is
GVHMR's own peak):

Figures are the GVHMR stage's own `[GPU] peak` meter, which is per-stage and more precise than
sampling `nvidia-smi`:

| Frames | GVHMR peak | GVHMR wall | Note |
|---:|---:|---:|---|
| 727 | **9.7 GB** | 86 s | rc=0 |
| 3,587 | **9.7 GB** | 334 s | rc=0 |
| 12,526 | **9.7 GB** | 1,130 s | rc=0 |
| 35,024 (20 min) | — | — | attempt FAILED before GVHMR finished; cause not captured |
| 35,755 (20 min) | 13,710 MiB | — | 2026-09, **pre-`ea4ba35`** — still the only long-clip figure |

**Flat at 9.7 GB from 727 to 12,526 frames** — three lengths, same number to the decimal. This
*supersedes* the older "13.7 GB at 12,520 frames": that sweep predates `ea4ba35`, which sizes
batches from free VRAM instead of total, and the same length now peaks ~4 GB lower.

**The 20-minute case is NOT re-measured.** A 2026-09-13 attempt at 35,024 frames failed 27 min in,
before GVHMR reported, and the log was lost to cleanup, so there is no cause on record — treat
that row as an open question, not as evidence of a problem. The pre-`ea4ba35` **13,710 MiB
remains the only long-clip figure**, and it is an upper bound: the banded-attention working set
does keep growing (3.78 GiB at 35,755 vs ~1.3 GiB at 12,526), so some rise above 12.5k is
expected. Re-running a 20-min clip is the single most useful missing measurement.

Budget **~14 GB to be safe at any length**; that bound is what the ~16 GB recommendation rests on,
and it is now conservative for anything up to ~12k frames, where 9.7 GB is the measured figure.

| you have | you can run | basis |
|---|---|---|
| 8 GB | up to 12,526 frames | MEASURED end-to-end to 369 frames (5.1 GB); INFERRED beyond, from the flat 9.7 GB GVHMR sweep and the fact that batches shrink to the card |
| ~16 GB | any length that fits in wall time | INFERRED — the 13.7 GB upper bound is the whole stage, but no 16 GB run of a 20-min video exists |
| 45 GB+ (rtx8000, a100, h100) | any length | MEASURED to 35,755 frames (pre-`ea4ba35`) |

`vid2smplx run` preflights this (`checks.py: vram_warning`) and warns — never refuses — and only
past 12,526 frames on a card under ~16 GB. It does **not** try to estimate your peak from the
frame count: peak follows the card, so that number would be fiction.

## Stage timings

### Quadro RTX 8000 (46 GB) — MEASURED 2026-09-13, gpu006, GVHMR `ea4ba35`, ViTPose fp16 OFF (the default)

| Stage | 39 frames | 150 frames | 369 frames | Peak VRAM @369 | GPU util |
|-------|----------:|-----------:|-----------:|---------------:|---------:|
| GVHMR (body) | 32.7 s | 41.7 s | 65.3 s | 10.0 GB | 10–32 % |
| HaMeR (hands) | 45.9 s | 64.2 s | 105.3 s | 8.3 GB | 6–19 % |
| EMICA (face) | 18.3 s | 25.3 s | 39.4 s | **12.3 GB** | 4–13 % |
| L2CS + MediaPipe | 10.1 s | 12.3 s | 18.1 s | 3.7 GB | 1–14 % |
| IK hands | 10.1 s | 9.2 s | 12.6 s | 1.3 GB | 10–48 % |
| Render (`--final-incam`) | 1.3 s | 1.4 s | 1.6 s | — | 0–3 % |
| **Total** | **123 s** | **159 s** | **246 s** | **12.3 GB** | |

Roughly **0.67 s per video frame** on this card. Note the peak stage at 369 frames is EMICA, not
GVHMR — again because the batch heuristic hands the free card to whichever stage asks.

### RTX PRO 1000 Laptop (8 GB, Blackwell), conda env `vid2smplx_bw`

MEASURED 2026-09, **before** the banded-attention and free-VRAM-batch changes. Kept because it is
the only 8 GB datapoint; treat the timings as historical, not current.

| Stage | 39 frames | 150 frames | 369 frames | Peak VRAM | GPU util |
|-------|----------:|-----------:|-----------:|----------:|---------:|
| GVHMR (body) | 44 s | 71 s | 123 s | 4.5 GB | 20–54 % |
| HaMeR (hands) | 63 s | 96 s | 167 s | 5.1 GB | 10–29 % |
| EMICA (face) | 28 s | 30 s | 59 s | 4.8 GB | 3–8 % |
| L2CS + MediaPipe | 6 s | 11 s | 18 s | 3.6 GB | 2–17 % |
| Merge | 2 s | 2 s | 2 s | — | — |
| IK hands | 11 s | 19 s | 36 s | 0.3 → 1.1 GB | 36–84 % |
| **Total** | **157 s** | **230 s** | **408 s** | **5.1 GB** | |

Roughly **1 s per video frame** on this laptop, ~5 min for a 15 s clip.

## What this means

- **The GPU is mostly idle.** Utilization is 1–48 %: time goes to video decoding, face/hand
  cropping and loading five models, not to the GPU. A bigger `--batch-size` will not speed this
  up; more CPU cores, a faster disk or a higher-clocked CPU will. `vid2smplx run` prints
  `[GPU] peak … util …%` after each stage and flags it as underutilized below 40 %.
- **Wall time scales linearly.** Budget ~1 s/frame on an 8 GB laptop, ~0.67 s/frame on an
  rtx8000. Splitting a video into chunks costs ~100 s of model reloads per chunk plus a seam in
  GVHMR's global trajectory, so the pipeline does not do it automatically — and since banded
  attention it does not need to.
- **Output is small.** MEASURED: 4.4 MB for a 39-frame clip, 16 MB for 150 frames with renders,
  46 MB for a 20-min clip after `--cleanup`.

## GVHMR VRAM was quadratic in frame count — fixed in GVHMR 8be5155

An earlier version of this page said "VRAM does not grow with clip length" and "long videos
are bounded by wall time and RAM, not VRAM". That was wrong, because GVHMR's HMR4D transformer
runs over the *whole* sequence and expressed its 120-frame attention window as a dense (L, L)
mask; `RoPEAttention` then materialised a full `(B, heads, L, L)` fp32 score tensor.

MEASURED on cleps with `P001_S01_BP.mp4` (1920×1080, 25 fps), on a Quadro RTX 8000.
**Taken before GVHMR `ea4ba35`**, when batches were sized against the card's TOTAL memory; with
free-VRAM sizing the short-clip rows come out lower (re-measured 2026-09-13: 9,963 MiB at 727
frames, not 13.7 GB). **Every 13.7 GB figure in the table below is therefore superseded** by the
9.7 GB sweep above for lengths up to 12,526 frames. Only the 35,755-frame row has not been
re-measured, so 13.7 GB survives solely as an **upper bound** at 20 min on a 46 GB card.

| Frames | GVHMR peak VRAM | Result |
|---:|---:|---|
| 721 | 13.7 GB (superseded: 9.7 GB post-`ea4ba35`) | rc=0 |
| 1,793 | 13.7 GB (superseded) | rc=0 |
| 3,581 | 13.7 GB (superseded: 9.7 GB at 3,587) | rc=0 |
| 7,157 | 13.7 GB (superseded) | rc=0 |
| 12,520 | 13.7 GB (superseded: 9.7 GB at 12,526) | rc=0 |
| 35,755 (20 min), **before** the fix | — | **OOM**: `Tried to allocate 38.10 GiB` on a Quadro RTX 8000 (45,364 MB), full pipeline, `RUN_EXIT=1` |
| **35,755** (20 min), **after** the fix | **13,710 MiB** — pre-`ea4ba35`, an **upper bound**, never re-measured | **rc=0** on the same Quadro RTX 8000, 1 h 18 m wall |

That 38.10 GiB was never the mask. The mask is bool and only 1.19 GiB at L=35,755, and
`expand` makes it a view, so it costs nothing extra. The 38.10 GiB is the *score* tensor:
35,755² × 8 heads × 4 bytes = 38.10 GiB, matching the OOM message to the last digit. Several
such tensors are needed (`masked_fill` twice, then `softmax`), so the true requirement was
≈76 GiB, which is why only an 80 GB card ever finished.

The window is now carried as per-query `[lo, hi)` bounds and attention runs blockwise through
`scaled_dot_product_attention`, reading only the keys each 2048-frame query block can see. The
score matrix is never built. Standalone, at the shapes GVHMR uses, on an rtx8000:

| Frames | dense einsum | banded blocks |
|---:|---:|---:|
| 16,000 | 15.60 GiB | 0.81 GiB |
| 35,755 | **OOM at 38.10 GiB** | **3.78 GiB** |

3.78 GiB sits under what the ViT stages already hold, so it no longer moves the peak at all —
hence a peak that is flat in clip length rather than quadratic (9.7 GB to 12,526 frames post-fix;
13,710 MiB at 35,755, pre-fix and not re-measured).

### How much the numbers move

Banded attention is the same operator, not an approximation, but it reassociates the
floating-point sums. Measured on the same GPU with the same cached features (so nothing but
the mask implementation differs), L = 12,000:

| quantity | dense vs banded, max abs | rms |
|---|---:|---:|
| `smpl_params_incam` `global_orient`, `transl` | 7.2e-7 | 1.0e-7 |
| `betas` | 1.2e-7 | 4.2e-8 |
| `body_pose` | 1.4e-3 | 4.6e-5 |
| `smpl_params_global` `global_orient` | 6.1 rad | 0.55 |

The last row is not the mask. World-frame orientation is produced by a **sequential product of
per-frame rotations** over the whole clip (`gvhmr_pipeline.py`), so any perturbation compounds:
the median error over the first 10 % of frames is 3.6e-7 and over the first 75 % it is 2.3e-3,
while the in-camera orientation stays flat at 9e-8 throughout. The pre-existing cross-machine
drift moves that same quantity by 6.16 rad, i.e. **world-frame `global_orient` on a 20-minute
clip was already not reproducible**, with or without this change. Use `*_incam` when you need
a reproducible orientation.

Against the H100 ground-truth run of the same 20-min video, the rtx8000 banded run differs by
`body_pose` rms 7.2e-2, `global_orient_incam` max 2.7e-2, `betas` max 1.5e-3. The control above
shows the mask contributes rms 4.6e-5 of that, so essentially all of it is the known
cross-machine drift plus different upstream detections (YOLO/ViTPose/HMR2 on Turing vs Hopper).
**Not measured:** the same video re-run with the dense mask on an H100, which would separate
"different GPU" from "different upstream detections". It needs a 3-day h100 queue slot.

**What this means for GPU choice** (see also `docs/CLEPS_SETUP.md` §2.3):

- **Measured, current:** 727 → 12,526 frames all peak at **9.7 GB** and return rc=0 on a 45 GB
  rtx8000 (post-`ea4ba35`).
- **Upper bound, not re-measured:** 35,755 frames peaked at 13,710 MiB on the same card *before*
  `ea4ba35`. The one post-fix attempt at 35,024 frames failed 27 min in and its log was lost, so
  there is **no post-fix 20-min measurement at all** — do not read the older rows as one.
- **Inferred, not measured:** a 16 GB card should handle a 20-min video, since 13.7 GB is the
  whole stage. No 16 GB run of a 20-min video exists.
- **Measured, historical:** h100 (80 GB) completes 20-min videos; 80 of the corpus logs are
  H100 NVL runs, made before this fix.

The practical rule: **8 GB up to 12,526 frames** (measured), **~16 GB beyond** (the 13.7 GB
upper bound plus headroom). `checks.py` warns on exactly that boundary and nowhere else —
peak follows the card, so frame count alone cannot predict it.

### Rejected alternatives

- **FlexAttention** (pass a mask *function*, never build one) is the textbook fix, but it needs
  torch ≥ 2.5 and this pipeline is pinned to **2.3.0** on both the workstation and cleps —
  `torch.nn.attention.flex_attention` is `ModuleNotFoundError` there. Verified on the cluster.
- **Chunk the video with overlap.** Viable but strictly worse. The receptive field is *not*
  ±60 frames: it is ±60 **per layer**, and there are 12 layers. Measured by perturbing one
  frame and counting which outputs change at all: ±60 after 1 layer, ±240 after 4, ±480 after
  8, **±720 after 12**. So exact interiors need ≥720 frames of overlap on each side, not 120.
  Worse, `avgbeta` averages `betas` over the *entire* sequence, so chunking changes `betas` for
  every frame no matter how large the overlap — no chunking is ever interior-exact. Banded
  attention is exact and needs no stitching, so this was not implemented.
- **Keep the mask boolean.** Already boolean, and already a view. Nothing to win.
- **Vectorise the 35,755-iteration Python mask loop.** Superseded: the (L, L) mask is not built
  at all any more. The bounds are computed with two `clamp`s.

## Cluster reference (from the README comparison, Quadro RTX 8000 46 GB)

| Clip | Frames | vid2smplx total | Notes |
|------|-------:|----------------:|-------|
| Talking 1080×1920 | 369 | 697 s | with final incam render |
| Dancing 1920×1080 | 598 | ~680 s | |
| Signing 4096×2160 | 478 | 1772 s | 4K is downscaled to 1080p first; HaMeR dominates |

## Four scheduling / throughput fixes (2026-09-13)

None of these change the algorithm. Three have no numerical effect at all; the fp16
one moves keypoints by ~1e-4 on heatmaps valued in ~[0, 1] and is validated below.

### 1. SLURM array concurrency: `%1` -> `%5`, `--time` 24 h -> 6 h

`--array=0-95%1` ran the corpus strictly one clip at a time, so it paid **96 sequential
queue waits**. Measured over 11,706 gpu-partition jobs: rtx8000 median 29 min / p90 15 h;
h100 median 34 min / p90 36 h / p99 144 h. Waiting, not computing, set the corpus wall
clock. QOS `normal` caps 8 concurrent GPUs, so `%5` leaves headroom.

The 24 h walltime was the second half of the problem: the backfill scheduler cannot slot a
24 h reservation into a gap, and the median clip is 1.4 h. 6 h is ~4x median headroom, and
overrunning is not data loss — `--requeue` plus temp-then-rename staging resumes at the
first unfinished stage.

Both are defaults, not policy: the submit line wins over an `#SBATCH` directive, so
`sbatch --array=0-$((N-1))%8 --time=12:00:00 scripts/slurm_interp.sh` overrides without
editing the file.

### 2. Auto batch size reads FREE VRAM, not TOTAL

Four near-identical copies of one heuristic (EMICA, HaMeR, ViTPose, HMR2-feature) all
budgeted against `get_device_properties(device).total_memory`. Three concurrent pipelines
therefore each printed `[Auto BS] EMICA: 47.6GB GPU -> bs=512` — each claiming the whole
card. Wall went 827 s -> >1500 s, **3.4x slower per clip**.

All four now call one helper, `GVHMR/hmr4d/utils/auto_batch.py`, which budgets against
`torch.cuda.mem_get_info()[0]`. Free bytes already net out the CUDA context, resident model
weights and every other process. `free <= total` always, so this cannot OOM anywhere the old
form fit. Measured on a 6 GB RTX 4050 with a small conv model:

| situation | old (total) | new (free) |
|---|---:|---:|
| alone on the card | 135 | 66 |
| 3.2 GB co-tenant resident | 50 | **1** |

Note the solo case also shrinks (135 -> 66). That is the point, not a regression: `total`
counted memory the model weights and CUDA context had already taken. Batch size has long
since hit diminishing returns at these values.

Per-stage caps, safety factors and the small-card (<12 GB) clamps are preserved as arguments
rather than four divergent hardcodings.

### 3. fp16 autocast on GVHMR's ViTPose

Measured on the card these jobs actually land on, **Quadro RTX 8000 (sm_7.5), gpu006**,
batch of 32:

| | ms / batch-32 |
|---|---:|
| fp32 | 754.2 |
| fp16 | 206.3 |
| **speedup** | **3.66x** |

Heatmap `max|d| = 1.9e-4` on outputs spanning `[-0.006, 0.081]`.

**fp16, never bf16.** bf16 measured 0.67x — *slower* — on sm_7.5, which has no bf16 tensor
cores. The repo already made this call once: `hamer/hand_detectors/vitpose_detect.py` runs
the same ViTPose architecture under `autocast(dtype=torch.float16)`.

Every forward routes through one `VitPoseExtractor._forward`, so the batch-size probe and the
real inference cannot drift to different dtypes. `VID2SMPLX_VITPOSE_FP16=0` restores fp32;
it exists as the control for the A/B below.

### 4. `seed_everything` no longer sets cudnn flags

`seed_everything` set `cudnn.deterministic = True` and `cudnn.benchmark = False`
process-wide. Because `scripts/utils.py` calls it at import time whenever `VID2SMPLX_SEED`
is set, every `--seed` run — which is how the functional tests execute — paid for it.
Measured on EMICA: **666 ms -> 1822 ms, 2.74x slower**.

These are backend performance flags, not RNG state, so they are simply gone; the function
seeds `random`, `numpy` and `torch` and nothing else. Unseeded runs were never affected.

The cost is card- and shape-dependent: a synthetic conv stack on an RTX 4050 (sm_8.9) shows
no difference at all (19.2 vs 19.1 ms), because there the deterministic algorithm *is* the
one autotuning would pick. The 2.74x above is the EMICA path on the cluster card.
