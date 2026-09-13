# Installation

## Install

```bash
git clone git@github.com:articulab/vid2smplx.git   # private repo: SSH. NOT --recursive
cd vid2smplx
bash install.sh                # conda env `vid2smplx`; use --uv for a repo-local .venv
conda activate vid2smplx       # or: source .venv/bin/activate
```

`install.sh` creates the env, initialises the submodules (public HTTPS — no key needed), downloads
the auto-downloadable weights and ends with `vid2smplx doctor`. Flags (`bash install.sh --help`):

| Flag / env var | Effect |
|---|---|
| `--uv` | `uv venv .venv` in the repo instead of a conda env |
| `--skip-models` | env only, no weights (`vid2smplx download` fetches them later) |
| `--env-only` | env + packages only: no submodules, no weights |
| `--force` | delete an existing conda env of this name and recreate it |
| `-h`, `--help` | show the usage message |
| `CONDA_ENV=name` | conda env name (default `vid2smplx`); ignored with `--uv` |
| `UV_HTTP_TIMEOUT=600` | seconds before a uv download times out |

Conda and uv install the same pinned packages.

## Then: the two licence-gated models

`doctor` ends with `[MISS]` for **SMPL-X** and **MANO**, which need a free registration and cannot
be downloaded automatically. Get them ([models.md](models.md) has the exact files), then:

```bash
vid2smplx doctor                                                     # every row [OK]?
vid2smplx run examples/clip_talking.mp4 --percent 10 --final-incam    # ~2 min smoke test
```

Result: `output/clip_talking_10pct/smplx_params.npz` + `render/final_incam.mp4`.

## Requirements

- Linux (tested on Ubuntu 20.04 / 22.04); git; conda or [uv](https://docs.astral.sh/uv/).
- NVIDIA GPU, **8 GB minimum**, **~16 GB** past 12,526 frames (~8 min at 25 fps). Peak follows the
  *card*, not the clip — the stages size their batches from free VRAM. Tested on RTX PRO 1000 8 GB,
  RTX 8000, A100, H100. Numbers and basis: [benchmarks.md](benchmarks.md).
- **~23 GB disk**: ~16 GB weights (15 GB in the repo + 1.4 GB in `~/.insightface`) + ~7 GB env.
- ffmpeg/ffprobe are **not** a prerequisite: `install.sh` vendors static builds into the env when
  they are not on PATH (the cluster case — no root, no module).

**Submodules.** GVHMR and HaMeR are our forks (public HTTPS, see `.gitmodules`); the changes
vid2smplx needs are committed there, so `git submodule update --init` delivers them — there is no
patch step. `install.sh` then runs `python3 -m vid2smplx.setup_submodules` (what `vid2smplx setup`
runs): it *verifies* GVHMR carries the required changes and aborts loudly otherwise, because without
the `--person` change a two-person clip would come back as a clean single-person SUCCESS. `doctor`
prints a `patch:` row per required change.

## Blackwell GPUs (RTX 50xx, RTX PRO, sm_120)

Stable PyTorch 2.3 has no sm_120 kernels. Use the separate env (PyTorch 2.10 + CUDA 12.8, pytorch3d
and nvdiffrast built from source, ~15 min):

```bash
bash specific_installation/install_blackwell.sh
conda activate vid2smplx_bw
CONDA_ENV=vid2smplx_bw vid2smplx doctor
```

If resolution fails, `pip install -r specific_installation/requirements_blackwell.txt` is the frozen
set. Differences: `TORCH_CUDA_ARCH_LIST="12.0"`, `setuptools<71` (mmcv needs `pkg_resources`), and
HaMeR loads lazily so 8 GB VRAM works.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `doctor` says `conda env 'vid2smplx' not found` | run `install.sh`, or `export CONDA_ENV=<name>` |
| `[MISS] link: ...` | `vid2smplx download` recreates the symlinks |
| `doctor` says `[CORRUPT] <model>` | truncated download: delete the file, re-run `vid2smplx download` |
| `doctor` says `[WARN] cuda ... below the documented minimum` | works on short clips; use `--percent` and `--batch-size 16` if it OOMs |
| CUDA out of memory in HaMeR | `--batch-size 16` |
| `IK_COVERAGE_LOW` | too few frames with visible hands; use another clip. `--no-hands` succeeds but the result is body+face only |
| `video not found` / `has no video stream` | checked with ffprobe before any model loads; run `ffprobe <file>` |
| Video > 1080p | downscaled automatically to `output/.downscaled/`; nothing to do |

## Tests

```bash
bash tests/run_tests.sh conda              # unit: CLI flags, doctor, docs (seconds, no GPU)
bash tests/run_tests.sh uv                 # same, inside .venv
bash tests/run_tests.sh conda functional   # real models on a 1.5 s clip (GPU, ~5 min)
```

160 unit tests (the 9 functional ones are deselected). The functional suite runs `vid2smplx run
--full-debug --gaze --seed 0` on the first 1.5 s of `examples/clip_talking.mp4` and checks every
stage — GVHMR shapes, HaMeR detections, EMICA/FLAME, gaze+blink, merged npz keys/shapes, IK coverage
and wrist error, renders — leaving a contact sheet, curves and the debug MP4s in
`tests/functional/out/` for eyeballing. Delete that directory to force a rerun.

**Reproducibility.** The seed makes the pipeline deterministic *on one machine* (4.5e-7 rad on
`body_pose` run to run, bit-identical elsewhere) but **not across machines**: a different
GPU/torch/CUDA build shifted GVHMR's output by ~0.027 rad on `body_pose` and ~0.088 on `betas` with
no code change — more than the tight tolerances (2 cm / 0.02 rad / 0.05 betas) allow. So each golden
stores its provenance (GPU, torch, CUDA, commit, dirty flag, submodule pointers) in the npz under
`__provenance__` and `test_golden` branches on it:

| golden produced… | behaviour |
| --- | --- |
| in this environment | tight tolerances — a regression fails |
| elsewhere, diff > ~2x the measured drift | **fails**, leading with the environment delta |
| elsewhere, diff within the drift | **skips** as inconclusive, naming the machine it came from |

One golden per (clip, length): `clip_talking_1.5s.npz` (38 frames, the default) and
`clip_dancing_20s.npz` (598 frames, `--clip examples/clip_dancing.mp4 --seconds 20`, which exercises
arms and IK). `bash tests/run_tests.sh conda functional --update-golden` accepts the current output
as the reference.
