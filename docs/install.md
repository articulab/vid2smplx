# Installation

> On the **cleps cluster**, follow [CLEPS_SETUP.md](CLEPS_SETUP.md) instead —
> the build must run on a GPU node and the licence-gated models are already staged there.

## Requirements

- Linux (tested on Ubuntu 20.04 / 22.04)
- NVIDIA GPU. **8 GB minimum**, enough for clips up to a few thousand frames; **~16 GB** for
  video beyond ~7 min (~12k frames). The stages size their batches from free VRAM, so the peak
  follows the card: measured 5.1 GB on an 8 GB RTX PRO 1000 and 12.3 GB on a 46 GB RTX 8000 for
  the same 369-frame clip. Tested on RTX PRO 1000 8 GB, RTX 8000, A100, H100. See
  [benchmarks.md](benchmarks.md).
- git, and either conda or [uv](https://docs.astral.sh/uv/). **ffmpeg/ffprobe are not a
  prerequisite**: `install.sh` vendors static builds of both into the env when they are not
  already on PATH (the cluster case — no root, no module).
- **~23 GB disk** for a complete install (measured). That is ~16 GB of weights — 15 GB inside the
  repo plus 1.4 GB in `~/.insightface`, which insightface hard-codes — and ~7 GB of Python env.

## Steps

```bash
git clone git@github.com:articulab/vid2smplx.git    # private: SSH, and NOT --recursive
cd vid2smplx
bash install.sh            # conda env `vid2smplx`, all deps, auto-downloadable weights
# or, without conda:
bash install.sh --uv       # uv-managed .venv in the repo (needs uv on PATH)
```

Both paths install the same pinned packages; the only difference is who owns the interpreter.
Afterwards `conda activate vid2smplx` or `source .venv/bin/activate` — the `vid2smplx` command
detects that it is inside the env and runs steps directly (outside any env it falls back to `conda run -n $CONDA_ENV`).

**Submodule changes.** GVHMR is our own fork (`https://github.com/MachtaYassine/GVHMR.git`, see `.gitmodules`). The changes vid2smplx needs from it — the `--person` rank and the track inventory that makes the multi-person guard work — are committed on that fork, so `git submodule update --init GVHMR` delivers them directly; there is no patch step. `install.sh` runs `python3 -m vid2smplx.setup_submodules` right AFTER the submodule update, and `vid2smplx setup` runs the same check by hand: it VERIFIES the checkout contains those changes and never modifies anything. A missing marker means the submodule is at the wrong commit — the message names the pinned commit and tells you to run `git submodule update --init GVHMR` (and that the pin is stale if that does not fix it). It aborts loudly rather than skipping: without the change a two-person clip would come back as a clean single-person SUCCESS. `vid2smplx doctor` prints a `patch:` row per required change.

`install.sh` ends by running `vid2smplx doctor`. It will report `[MISS]` for the two files that need
a (free) registration: **SMPL-X** and **MANO**. Follow the links it prints — details in [models.md](models.md) — then:

```bash
conda activate vid2smplx
vid2smplx doctor                                        # everything [OK]?
vid2smplx run examples/clip_talking.mp4 --percent 10 --final-incam   # ~2 min smoke test
```

Output lands in `output/clip_talking_10pct/` with `smplx_params.npz` and `render/final_incam.mp4`.

### install.sh flags

| Flag | Effect |
|------|--------|
| `--uv` | use `uv venv .venv` instead of conda |
| `CONDA_ENV=name bash install.sh` | conda env name (default `vid2smplx`) |
| `--skip-models` | don't download weights (run `vid2smplx download` later) |
| `--env-only` | create env + packages only |
| `--force` | delete and recreate the conda env |

## Blackwell GPUs (RTX 50xx, RTX PRO, sm_120)

Stable PyTorch 2.3 has no sm_120 kernels. Use the separate env (PyTorch 2.10 + CUDA 12.8, pytorch3d and nvdiffrast built from source, ~15 min):

```bash
bash specific_installation/install_blackwell.sh
conda activate vid2smplx_bw
CONDA_ENV=vid2smplx_bw vid2smplx doctor
```

If dependency resolution fails, `pip install -r specific_installation/requirements_blackwell.txt` gives the frozen set.
Differences vs the main env: `TORCH_CUDA_ARCH_LIST="12.0"`, `setuptools<71` (mmcv needs `pkg_resources`), and HaMeR loads models lazily so 8 GB VRAM works.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `doctor` says `conda env 'vid2smplx' not found` | run `install.sh`, or `export CONDA_ENV=<name>` if you used another env name |
| `[MISS] link: ...` | `vid2smplx download` recreates the symlinks |
| `L2CSNet_gaze360.pkl` missing | `vid2smplx download` re-fetches it from the HuggingFace mirror (`ymachta/articumotion-checkpoints`) and verifies its sha256 |
| `doctor` says `[CORRUPT] <model>` | the file is smaller than the real weights (interrupted download, full disk, or a truncated download saved as `.pkl`); delete it and re-run `vid2smplx download` |
| `doctor` says `[WARN] cuda ... below the documented minimum` | the run still works on short clips; use `--percent` and `--batch-size 16` if it OOMs |
| CUDA out of memory in HaMeR | `--batch-size 16` |
| IK step exits with `IK_COVERAGE_LOW` | too few frames with visible hands; prefer a different clip. `--no-hands` also succeeds, but the result has no hand poses at all — the params are body+face only |
| `video not found` / `has no video stream` | the input is checked with ffprobe before any model loads; run `ffprobe <file>` to see what it is |
| Video > 1080p | it is downscaled automatically to `output/.downscaled/`; nothing to do |

## Tests

```bash
bash tests/run_tests.sh conda              # unit tests: CLI flags, doctor, docs (seconds, no GPU)
bash tests/run_tests.sh uv                 # same, inside .venv
bash tests/run_tests.sh conda functional   # real models on a 1.5 s seeded clip (GPU, ~5 min)
```

The functional suite runs `vid2smplx run --seed 0 --full-debug` on the first 1.5 s of `examples/clip_talking.mp4`
and checks every stage: GVHMR shapes, HaMeR detections, EMICA/FLAME, gaze+blink, merged npz keys/shapes,
IK coverage and wrist error, renders. It writes `tests/functional/out/contact_sheet.png`, `curves.png`
and the debug MP4s for eyeballing.

Reproducibility: the seed makes the pipeline deterministic *on one machine* — two runs measured 4.5e-7 rad
apart on `body_pose`, bit-identical elsewhere. Across machines it is not: a different GPU / torch / CUDA build
shifted GVHMR's raw output by ~0.027 rad on `body_pose` and ~0.088 on `betas` with no code change at all,
which is more than the tight tolerances (2 cm / 0.02 rad / 0.05 betas) allow.

So the golden carries its provenance (GPU, torch, CUDA, commit, dirty flag, submodule pointers), stamped into
the npz under `__provenance__` by `--update-golden`, and `test_golden` branches on it:

| golden produced… | behaviour |
| --- | --- |
| in this environment | tight tolerances, no excuses — a regression fails |
| elsewhere, diff > ~2x the measured drift | **fails**, message leads with the environment delta, then the numbers |
| elsewhere, diff within the drift | **skips** as inconclusive, telling you which machine it came from |

A cross-machine run therefore never silently passes and never fails mysteriously. The first run skips the test;
once the visuals look right, accept the output as the reference:

```bash
bash tests/run_tests.sh conda functional --update-golden
```

Options: `--seconds 3` (longer clip), `--clip examples/clip_signing.mp4`. Delete `tests/functional/out/` to force a rerun.
