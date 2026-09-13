"""Provenance for the functional golden reference.

Why this exists
---------------
The functional goldens (``tests/functional/golden/clip_talking_1.5s.npz``, 38 frames, the
default; and ``tests/functional/golden/clip_dancing_20s.npz``, 598 frames, opt-in) are compared
against a fresh run with tight tolerances (~2e-2 rad). Measured on this repo:

* two runs on the SAME machine agree to 4.5e-7 rad on ``body_pose`` and are bit-identical
  on every other key — the pipeline is deterministic once seeded;
* a run on a DIFFERENT machine (or a different torch/CUDA build) drifted GVHMR's raw output
  by ~0.027 rad on non-arm ``body_pose`` joints and ~0.088 on ``betas`` with no code change.

0.027 > 2e-2 and 0.088 > 5e-2, so the tight tolerances turn into false failures the moment
the golden travels. Widening them to swallow that would also swallow real regressions (the
un-IK'd-arms bug this test caught was 0.54 rad on the arms, but hand/gaze keys are checked at
5e-2 — an order of magnitude below the betas drift).

So instead of one loose number we record WHERE the golden was produced and branch on it.
Everything here is pure logic apart from :func:`capture_environment`, so it is unit-testable
without a GPU.
"""
from __future__ import annotations

import functools
import json
import platform
import subprocess
from pathlib import Path

# Key under which the JSON blob is stored inside the golden .npz.
PROVENANCE_KEY = "__provenance__"

# Fields that must match for the tight tolerances to be considered meaningful.
# Ordered most- to least-likely to be the cause of a numeric difference.
ENV_FIELDS = ("gpu", "torch", "cuda")
# Recorded and reported, but a difference here does not by itself relax anything:
# the code identity is what the test is *for*, so a code change must still fail loudly.
CODE_FIELDS = ("git_commit", "git_dirty")


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def capture_environment(repo: Path | None = None) -> dict:
    """Describe the machine + code that is producing (or checking) a golden.

    Never raises: a missing torch or a non-git checkout degrades to "unknown" strings, because
    failing to *record* provenance must not fail the run that was otherwise fine.
    """
    repo = Path(repo) if repo is not None else Path(__file__).resolve().parents[1]
    env = {
        "gpu": "unknown",
        "torch": "unknown",
        "cuda": "unknown",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git(repo, "rev-parse", "HEAD") or "unknown",
        "git_dirty": bool(_git(repo, "status", "--porcelain")),
        "git_submodules": _git(repo, "submodule", "status"),
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda"] = str(torch.version.cuda)
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return env


@functools.lru_cache(maxsize=4)
def code_identity(repo: str | None = None) -> tuple[tuple[str, object], ...]:
    """Which code produced an artifact: repo HEAD + every submodule pin, as a hashable pair list.

    Stage stamps key on inputs and flags only, so after a `git pull` that changes code or bumps
    a submodule, a re-run into an existing output dir reported [SKIP] and kept results from the
    OLD code. `git_dirty` is a flag, not a hash: two different dirty trees at one commit look
    the same, which is the price of not hashing the worktree on every run.
    """
    env = capture_environment(Path(repo) if repo else None)
    return tuple((k, env[k]) for k in ("git_commit", "git_dirty", "git_submodules"))


def golden_unusable(golden: dict | None) -> str:
    """Why this golden cannot support a regression check at all ('' when it can).

    A golden that records nothing about where it came from used to be routed through
    :func:`compare_environment`, which reported a full 3-field ENVIRONMENT mismatch. That
    made :func:`verdict` unable to return anything but 'fail' (only over the 3x-looser
    cross-env envelope) or 'skip' -- so a real 0.05 rad regression, five times the tight
    tolerance, came out as a *skip*: green, and invisible unless pytest ran with -rs.

    A regression check that cannot fail is worse than no check, because it is believed.
    So this is a hard failure that names its own remedy, not a skip. The alternative --
    keep skipping and merely print louder -- was rejected: CI reads exit codes, not text.
    """
    if golden:
        return ""
    return ("This golden carries NO provenance (it predates the __provenance__ member), so there is\n"
            "no way to tell whether the tight tolerances mean anything for it, and the comparison\n"
            "below could not fail no matter how large a regression it saw.\n"
            "Regenerate it: `bash tests/run_tests.sh uv functional --update-golden`\n"
            "(eyeball tests/functional/out/ first — --update-golden accepts whatever it finds).")


def compare_environment(golden: dict | None, current: dict) -> list[str]:
    """Return a human-readable list of ENV_FIELDS that differ. Empty list == same environment.

    A golden with no provenance at all (``None``) counts as a full mismatch: we cannot claim
    the tight tolerances are meaningful for a reference of unknown origin. Callers must run
    :func:`golden_unusable` FIRST and stop there — a full mismatch reported from here can only
    ever produce 'fail' or 'skip', never 'pass', which silently disables the check.
    """
    if not golden:
        return [f"{f}: golden records nothing (predates provenance) vs now {current.get(f, 'unknown')}"
                for f in ENV_FIELDS]
    out = []
    for f in ENV_FIELDS:
        g, c = golden.get(f, "unknown"), current.get(f, "unknown")
        if g != c:
            out.append(f"{f}: golden {g!r} vs now {c!r}")
    return out


def code_differences(golden: dict | None, current: dict) -> list[str]:
    """Differences in the code identity, reported to make a failure diagnosable."""
    if not golden:
        return []
    out = []
    if golden.get("git_commit") != current.get("git_commit"):
        out.append(f"git_commit: golden {str(golden.get('git_commit'))[:12]} vs now "
                   f"{str(current.get('git_commit'))[:12]}")
    if golden.get("git_dirty") or current.get("git_dirty"):
        out.append(f"git_dirty: golden {bool(golden.get('git_dirty'))} vs now "
                   f"{bool(current.get('git_dirty'))}")
    if golden.get("git_submodules") != current.get("git_submodules"):
        out.append("git_submodules: submodule pointers differ (GVHMR/HaMeR/inferno)")
    return out


def describe(env: dict | None) -> str:
    if not env:
        return "<no provenance recorded>"
    return (f"GPU={env.get('gpu', '?')} torch={env.get('torch', '?')} cuda={env.get('cuda', '?')} "
            f"commit={str(env.get('git_commit', '?'))[:12]}{'+dirty' if env.get('git_dirty') else ''}")


def mismatch_message(golden: dict | None, current: dict, env_diffs: list[str],
                     failures: list[str], *, over_envelope: bool) -> str:
    """The message a cross-environment comparison prints. Environment first, numbers second.

    ``over_envelope`` distinguishes "bigger than anything cross-machine drift has been measured
    to cause" (a real regression, reported as such) from "inside the measured drift" (not
    evidence of anything, reported as inconclusive).
    """
    head = ("This golden was NOT produced in the current environment.\n"
            f"  golden : {describe(golden)}\n"
            f"  current: {describe(current)}\n"
            "  differs on: " + "; ".join(env_diffs) + "\n"
            "A cross-environment offset of ~0.027 rad on body_pose and ~0.088 on betas has been\n"
            "measured on this pipeline with no code change at all, so differences of that size\n"
            "here are expected and prove nothing.\n")
    code = code_differences(golden, current)
    if code:
        head += "  NOTE: the code also differs — " + "; ".join(code) + "\n"
    if over_envelope:
        body = ("These differences are LARGER than the measured cross-environment drift, so they\n"
                "are unlikely to be explained by hardware alone — treat this as a real regression:\n")
    else:
        body = ("All differences are within the measured cross-environment envelope — inconclusive:\n")
    tail = ("\nTo get a meaningful check: rerun on the machine that produced the golden, or\n"
            "regenerate here with `bash tests/run_tests.sh uv functional --update-golden`\n"
            "(only do that if you have eyeballed tests/functional/out/ and the output is good).")
    return head + body + "\n".join("  " + f for f in failures) + tail


def verdict(env_diffs: list[str], bad_tight: list[str], bad_wide: list[str]) -> str:
    """'pass' | 'fail' | 'skip' — the whole policy in one place.

    Same environment: tight tolerance decides, no excuses. Different environment: only a
    difference bigger than the measured cross-machine drift fails; anything smaller is
    reported as inconclusive (skip), never as a silent pass.
    """
    if not env_diffs:
        return "fail" if bad_tight else "pass"
    if bad_wide:
        return "fail"
    return "skip"


def read_provenance(npz) -> dict | None:
    """Pull the provenance dict out of a loaded npz mapping (or ``None`` if absent/corrupt)."""
    if PROVENANCE_KEY not in npz:
        return None
    try:
        return json.loads(str(npz[PROVENANCE_KEY]))
    except Exception:
        return None


def write_golden(src: Path, dst: Path, env: dict) -> None:
    """Copy a run's smplx_params.npz to the golden path, stamping the environment into it."""
    import numpy as np

    data = dict(np.load(src, allow_pickle=True))
    data[PROVENANCE_KEY] = np.str_(json.dumps(env, sort_keys=True))
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **data)
