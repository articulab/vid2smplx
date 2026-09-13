# vid2smplx on cleps — first setup, then one-shot

Written from an actual from-scratch bringup (2026-09-13). Every command below was
run; the numbers and paths are measured, not assumed.

Two parts: **§1 you do once per machine** (needs a GPU node, ~90 min, mostly waiting),
**§2 is one command per corpus** thereafter.

---

## 0. What you need before starting

| | |
|---|---|
| cleps account + SSH | `ssh cleps` |
| GitHub access to `articulab/*` | SSH **agent forwarding** — see §1.1 |
| SMPL-X + MANO licences | staged on cleps under **`ymachta`** at `/scratch/ymachta/Corpus/` — §1.3 |
| free scratch | **~25 GB** for the install (measured 23 GB); leave 30 GB of headroom, plus your corpus |

---

## 1. First setup (once)

### 1.1 Log in with agent forwarding

The repos are private and cleps has **no deploy key**. Your local SSH agent must be
forwarded, or every `git clone` fails with `Repository not found`:

```bash
ssh -A cleps          # -A is required; put `ForwardAgent yes` in your ~/.ssh/config
ssh-add -l            # on cleps: must list your key(s)
```

### 1.2 Clone into scratch, not home

`/scratch` is the only filesystem with room; home is small and shared.

```bash
mkdir -p /scratch/$USER/work && cd /scratch/$USER/work
git clone git@github.com:articulab/vid2smplx.git
cd vid2smplx
```

### 1.3 Link the licence-gated body models

These cannot be downloaded automatically (Max-Planck registration). They are
already on cleps, so link rather than re-download:

These are **`ymachta`'s** copies, staged under that account — not `$USER`. Point at them
directly, or at your own copy if you have registered and downloaded your own:

```bash
mkdir -p models
ln -sfn /scratch/ymachta/Corpus/smplx            models/smplx   # or your own SMPL-X v1.1 dir
ln -sfn /scratch/ymachta/Corpus/mano_v1_2/models models/mano    # or your own MANO v1.2 dir
ls -L models/smplx models/mano                                  # MUST list files, not error
```

`ls -L` follows the link: if it prints `No such file or directory` the symlink is dangling
and every later step fails confusingly. `models/smplx` must contain `SMPLX_NEUTRAL.npz`
**and** `MANO_SMPLX_vertex_ids.pkl` (the IK stage hard-fails without the second one;
`doctor` checks for both). If you cannot read `/scratch/ymachta/Corpus`, register at
https://smpl-x.is.tue.mpg.de/ and https://mano.is.tue.mpg.de/ — see [models.md](models.md).

### 1.4 Build on a GPU node — never on the login node

`pytorch3d`, `detectron2` and `xtcocotools` compile CUDA extensions, so the build
needs a GPU *and* `nvcc`. The login node has neither and compiling there is
forbidden. Submit it as a batch job:

```bash
cat > build.sbatch <<'SB'
#!/bin/bash
#SBATCH --job-name=v2s_build --partition=gpu --gres=gpu:1
#SBATCH --time=03:00:00 --cpus-per-task=16 --mem=64G
#SBATCH --output=%x_%j.log
export CUDA_HOME=${CUDA_HOME:-/scratch/$USER/miniconda3}   # nvcc 12.5 lives here
export MAX_JOBS=16
cd /scratch/$USER/work/vid2smplx
bash install.sh --uv
SB
sbatch build.sbatch
squeue -u $USER          # watch it
```

Measured (2026-09-13, rtx8000): **21 minutes**, **23 GB on disk** — 7.4 GB `.venv`,
15 GB of weights across `models/`, `GVHMR/inputs/`, `hamer/_DATA/`, plus the submodule
sources. A further 1.4 GB lands in `~/.insightface`, which is on **home**, not scratch.
Outputs are small by comparison: ~5-15 MB per short clip, 46 MB for a 20-min clip after
`--cleanup`.

`CUDA_HOME` matters: there is no `cuda` module on cleps and no system `nvcc` — the
only toolkit is the one inside your miniconda (`nvcc 12.5`). Without it the CUDA
extensions fail to compile.

**`--uv` is not optional.** cleps auto-activates conda `base` on every node, and
without `--uv` the installer would install into that shared env — mutating the
environment every other workstream uses. `--uv` builds an isolated `.venv`
(CPython 3.10.21, the version this stack is pinned to).

### 1.5 ffmpeg / ffprobe — handled for you

Cluster nodes have neither on PATH, no module provides them, and you have no root.
`install.sh` detects this and vendors both from `static-ffmpeg` into `.venv/bin`, so
the env is self-contained. Nothing to do — just don't be surprised by the log line.

(If you ever see `FileNotFoundError: 'ffprobe'`, that vendoring did not run: re-run
`install.sh --uv`.)

### 1.6 Verify

```bash
source .venv/bin/activate
vid2smplx doctor           # every line must read [OK]
```

`doctor` checks ffmpeg, ffprobe, every import, CUDA, all weights and the submodule
symlinks. If anything says `[MISS]`, fix that line and re-run — the installer is
idempotent.

Verified on a fresh clone (2026-09-13, gpu014 / RTX 2080 Ti):

```
All checks passed.
hands_left 100%  hands_right 100%  face 100%  gaze 100%  ik_coverage 97.4%
```

---

## 2. Running (one-shot, every time after)

### 2.1 One video

```bash
srun --partition=gpu --gres=gpu:1 --time=2:00:00 --mem=64G \
  bash -c "cd /scratch/$USER/work/vid2smplx && source .venv/bin/activate && \
           vid2smplx run /path/to/clip.mp4 --final-incam"
```

Output lands in `output/<clip-name>/`: `smplx_params.npz`, `gaze_blink/`, `render/`.

### 2.2 A whole folder — the SLURM array

Point it at a directory and it fans one video per task across the cluster. Use
`scripts/slurm_interp.sh`; `LIST` and `OUT` are the two knobs.

```bash
cd /scratch/$USER/work/vid2smplx          # sbatch must run from the checkout
IN=/scratch/$USER/my_corpus
OUT=/scratch/$USER/my_corpus_out
mkdir -p "$OUT"
find "$IN" -name '*.mp4' | sort > "$OUT/videos.txt"

sbatch --array=0-$(($(wc -l < "$OUT/videos.txt")-1))%4 \
       --export=ALL,LIST="$OUT/videos.txt",OUT="$OUT" \
       --output="$OUT/logs/%A_%a.out" --error="$OUT/logs/%A_%a.out" \
       scripts/slurm_interp.sh
```

`%4` caps concurrency at 4 tasks. Results land in `$OUT/<clip-name>/`.

The script's own defaults are `%5` and `--time=06:00:00`; the submit line always wins, so
override either here rather than editing the file. `%1` is a trap — it makes an N-clip
corpus pay N *sequential* queue waits (median 29 min on rtx8000), which dominates the
compute. QOS `normal` caps you at 8 concurrent GPUs.

`#SBATCH` directives are read before any shell runs, so they cannot mention `$OUT`. Logs
therefore default to the directory you run `sbatch` from; the `--output`/`--error` above
move them next to the results (the script does `mkdir -p "$OUT/logs"`, so create that dir
first if you use it: `mkdir -p "$OUT/logs"`). Same rule for the GPU pick: the script
defaults to `--constraint="rtx8000|a100|h100|h200" --exclude=gpu018`. Prefer rtx8000 — see
§2.3 for why.

Three things it does for you:

- **Idempotent** — each finished clip gets a `DONE` marker and is skipped on
  resubmit, so you can just resubmit a partially-finished array.
- **Quality-gated** — a clip whose hand coverage is under 5% is marked `FAILED`
  rather than counted as success ("no hands" and "HaMeR broke" are different things).
- **Disk-guarded** — refuses to start below 200 GB free on scratch.

**Run `sbatch` from the checkout.** SLURM spools the script to its own directory, so
the script locates the repo via `SLURM_SUBMIT_DIR`. If you must submit from
elsewhere, pass `REPO=/path/to/vid2smplx` in `--export`.

Verified 2026-09-13: a 2-clip array across gpu007 + gpu014 finished with hand
coverage 0.962 / 0.983 and wrote both `DONE` markers.

### 2.3 Pick the GPU by video length

GVHMR's HMR4D transformer used to express its 120-frame attention window as a dense (L, L)
mask, so VRAM grew as O(frames²) and only an 80 GB card finished a 20-min video. **GVHMR
`8be5155` removed that**: the window is banded and the score matrix is never built. Full
numbers and the numerical control are in [benchmarks.md](benchmarks.md); the operational
summary is this table.

| video length | `--constraint` | basis |
|---|---|---|
| up to ~12k frames (7 min) | `rtx8000\|a100` | MEASURED, rc=0 |
| ~35k frames (20 min) | `rtx8000\|a100\|h100\|h200` (~16 GB is enough) | MEASURED on rtx8000: 13,710 MiB, 1 h 18 m, rc=0 |

**Prefer `rtx8000`.** It allocates in minutes; h100 is a multi-day queue (median 34 min but
p90 36 h, over 11,706 gpu-partition jobs). Since the fix there is no reason to wait for an
h100 — GVHMR measured **9.7 GB, flat from 727 to 12,526 frames**, on an rtx8000 (2026-09-13,
GVHMR `ea4ba35`). At 20 min the only figure is the pre-`ea4ba35` 13,710 MiB upper bound.

⚠️ The corpus in `/scratch/ymachta/interpersonality_out` (101 entries) was produced with the
old dense mask on H100. Mixing old and new outputs is fine for `*_incam` quantities (they
agree to 7e-7) but **not** for world-frame `global_orient`, which is not reproducible across
either change — see benchmarks.md.

cleps inventory: `gpu002-003` rtx6000 · `gpu006-009` rtx8000 · `gpu011,014` rtx2080ti
· `gpu012-013` a100 · `gpu015-017` h100 · `gpu018` h200.

---

## 3. Things that will bite you

- **`git clone` fails** → you forgot `ssh -A`. cleps has no deploy key of its own.
- **Installed into conda base** → you omitted `--uv`.
- **IK stage fails** → `MANO_SMPLX_vertex_ids.pkl` missing from `models/smplx/`.
- **Hands 0% but no error** → the HaMeR data symlink was shadowed by an empty
  directory. `vid2smplx doctor` now reports this as `[MISS]` instead of `[OK]`.
- **Job dies at ~90% on a long video** → VRAM. See §2.3.
- **`/scratch` full** → the array refuses to start below 200 GB free, by design.
