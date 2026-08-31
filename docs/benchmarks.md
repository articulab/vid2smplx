# Benchmarks

Measured with `python scripts/bench.py examples/clip_talking.mp4 --percent 10 40 100` (1080×1920 portrait, 25 fps).
Peak VRAM is above the idle baseline; utilization is the mean `nvidia-smi` GPU-util during the stage.
Raw numbers: [benchmarks.json](benchmarks.json).

## RTX PRO 1000 Laptop (8 GB, Blackwell), conda env `vid2smplx_bw`

| Stage | 39 frames | 150 frames | 369 frames | Peak VRAM | GPU util |
|-------|----------:|-----------:|-----------:|----------:|---------:|
| GVHMR (body) | 44 s | 71 s | 123 s | 4.5 GB | 20–54 % |
| HaMeR (hands) | 63 s | 96 s | 167 s | 5.1 GB | 10–29 % |
| EMICA (face) | 28 s | 30 s | 59 s | 4.8 GB | 3–8 % |
| L2CS + MediaPipe | 6 s | 11 s | 18 s | 3.6 GB | 2–17 % |
| Merge | 2 s | 2 s | 2 s | — | — |
| IK hands | 11 s | 19 s | 36 s | 0.3 → 1.1 GB | 36–84 % |
| **Total** | **157 s** | **230 s** | **408 s** | **5.1 GB** | |

Fit: `GVHMR ≈ 33 s + 0.24 s/frame`, `HaMeR ≈ 52 s + 0.31 s/frame`, `EMICA ≈ 19 s + 0.09 s/frame`.
Roughly **1 s of processing per video frame** on this laptop, ~5 min for a 15 s clip. Renders not included (`--final-incam` adds ~15 s per 40 frames).

## What this means

- **VRAM does not grow with clip length** for the four neural stages — each streams frames in batches. A 6 GB GPU runs the pipeline; 8 GB is comfortable. The only stage that scaled was IK (≈3 MB/frame); its chunk size is now derived from free VRAM (`ik_hands.py`), so it stays flat too.
- **The GPU is mostly idle.** Utilization is 3–54 %: time goes to video decoding, face/hand cropping and loading five models, not to the GPU. A bigger `--batch-size` will not speed things up on this machine; more CPU cores, faster disk, or a higher-clocked CPU will. `vid2smplx run` prints `[GPU] peak … util …%` after each stage and flags it as underutilized below 40 %.
- **Long videos** are therefore bounded by wall time and RAM, not VRAM. Budget ~1 s/frame on a laptop, ~0.5 s/frame on an A100 (see the README timing table). Splitting a video into chunks does not help VRAM and costs ~100 s of model reloads per chunk plus a seam in GVHMR's global trajectory, so the pipeline does not do it automatically. If you must run a multi-hour recording, cut it at natural scene boundaries and process the pieces independently.

## Cluster reference (from the README comparison, Quadro RTX 8000 46 GB)

| Clip | Frames | vid2smplx total | Notes |
|------|-------:|----------------:|-------|
| Talking 1080×1920 | 369 | 697 s | with final incam render |
| Dancing 1920×1080 | 598 | ~680 s | |
| Signing 4096×2160 | 478 | 1772 s | 4K is downscaled to 1080p first; HaMeR dominates |
