# Installation

## Requirements

- Linux (tested on Ubuntu 20.04 / 22.04)
- NVIDIA GPU, 8 GB+ VRAM (5.1 GB peak measured; tested on RTX PRO 1000 8 GB, RTX 8000, A100, H100)
- Conda, git, ffmpeg
- ~15 GB disk for weights

## Steps

```bash
git clone https://github.com/articulab/vid2smplx.git
cd vid2smplx
bash install.sh            # conda env `vid2smplx`, all deps, auto-downloadable weights
# or, without conda:
bash install.sh --uv       # uv-managed .venv in the repo (needs ffmpeg + uv on PATH)
```

Clone **without** `--recursive`: `inferno` declares nested submodules on a private GitLab
(`rdanecek/infernal_sandbox`, `rdanecek/infernal_apps`) that no one outside the author can read, so
`--recursive` aborts the whole clone with `Host key verification failed`. `install.sh` Phase 1 does
the submodule init properly — full recursion for GVHMR and HaMeR, and only the six `external/*`
submodules inferno actually needs.

Both paths install the same pinned packages; the only difference is who owns the interpreter.
Afterwards `conda activate vid2smplx` or `source .venv/bin/activate` — the `vid2smplx` command
detects that it is inside the env and runs steps directly (outside any env it falls back to `conda run -n $CONDA_ENV`).

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

## SLURM clusters

On a shared cluster the install has to run as a batch job on a GPU node, and ffmpeg/uv
are usually absent. See [cleps.md](cleps.md).

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
| `L2CSNet_gaze360.pkl` missing | gdown is rate-limited; download from the Drive link by hand into `models/` |
| CUDA out of memory in HaMeR | `--batch-size 16` |
| IK step exits with `IK_COVERAGE_LOW` | too few frames with visible hands; rerun with `--no-hands` or a different clip |
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

Reproducibility: the seed fixes every CPU/torch RNG, but CUDA kernels are not bit-exact across runs or GPUs,
so `test_golden` compares against `tests/functional/golden/smplx_params.npz` with tolerances (2 cm / 0.02 rad).
The first run skips that test; once the visuals look right, accept the output as the reference:

```bash
bash tests/run_tests.sh conda functional --update-golden
```

Options: `--seconds 3` (longer clip), `--clip examples/clip_signing.mp4`. Delete `tests/functional/out/` to force a rerun.
