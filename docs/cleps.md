# Setting up on cleps (or any SLURM cluster)

Written by doing it: the tree was deleted and rebuilt from an empty clone, and every
step below is one that actually bit. The generic instructions are in [install.md](install.md);
this page is only the cluster-specific parts.

## The short version

```bash
# on the login node — cloning and submitting only
git clone https://github.com/articulab/vid2smplx.git   # NOT --recursive, see install.md
cd vid2smplx

# everything else runs on a compute node
sbatch --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=06:00:00 \
       --wrap 'source ~/miniconda3/etc/profile.d/conda.sh && bash install.sh'
```

Then put the registration-gated files in place (see [models.md](models.md)) and run
`vid2smplx doctor` — also on a compute node, and inside the env:

```bash
conda activate vid2smplx     # the CLI is installed into the env, not on PATH
vid2smplx doctor
```

## Why install has to be a batch job

`install.sh` runs `pip` and `conda`, i.e. Python. Login nodes are for editing and
submitting, not for compute, and running Python there is blocked outright on cleps.
`conda create` alone pegs a core for minutes.

It also needs a GPU, because Phase 6 verifies CUDA and the import of every backbone.
Installing on a CPU node produces an env that reports fine and then fails on the first
real run.

Budget ~30 min with model downloads, ~5 min with `--skip-models`.

## Partition and QOS

```
sbatch: error: Job's QOS not permitted to use this partition
        (almanach allows proprietary not normal)
```

Partitions are gated on QOS, and the two are checked independently. Check what you have:

```bash
sacctmgr -n show assoc user=$USER format=Partition,QOS%40
```

A plain `normal` QOS means the `gpu` partition, not `almanach`. `sinfo -o "%20N %10P %.6t %30G"`
lists which GPUs sit where.

**Ask for the smallest GPU that fits.** The install and the test suite need almost no
VRAM and schedule instantly on an idle V100; a `--constraint=h100` on that job just
queues behind real work. Full-length 20-minute clips are the opposite — those need
h100/h200, since an A100 runs out of memory (~38 GiB peak).

## ffmpeg and uv are not installed

Neither is on `PATH`, and there is no module for either.

- **conda path**: nothing to do, `install.sh` installs ffmpeg from conda-forge into the env.
- **uv path**: `install.sh --uv` only warns, and its suggestion (`apt install ffmpeg`) needs
  root you do not have. Either borrow the conda env's binary
  (`export PATH=~/miniconda3/envs/vid2smplx/bin:$PATH`) or drop a static build in `~/.local/bin`.

`uv` itself installs per-user without root:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # -> ~/.local/bin/uv
```

Measured on cleps, env + packages only: **conda ~22 min, uv ~4.5 min**. Both produce an
env that passes the unit tests. Model downloads (~15 GB) are the same either way.

## Weights: keep them outside the clone

The two registration-gated downloads (SMPL-X, MANO) cannot be re-fetched by any script.
Keep one copy outside the repo and seed from it, so a clone can be deleted freely:

```
/scratch/$USER/model_zoo/
├── smplx/SMPLX_NEUTRAL.npz
├── smplx/MANO_SMPLX_vertex_ids.pkl      # without this the IK stage silently no-ops
├── mano/MANO_{LEFT,RIGHT}.pkl
└── L2CSNet_gaze360.pkl                  # gdown is rate-limited often enough to matter
```

`cp -a` them into `models/` after installing, then `vid2smplx download` to link the rest.

## Scratch

`/scratch` is Lustre and shared. `du` over a large tree takes many minutes and is unkind
to the metadata servers; `lfs quota -h -u $USER /scratch` answers the same question instantly.

Many small files are the slow case, which is why `conda create` and submodule checkouts
take longer here than on a laptop.

## Checking a run

`squeue`/`sacct` for state, but for correctness read the log: the pipeline prints `[OK]`
or `[FAIL]` per stage and writes a `FAILED` marker next to the output when a quality gate
trips, so a job that exits 0 is not by itself evidence the clip is usable.
