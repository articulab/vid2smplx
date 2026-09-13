"""The golden's provenance logic: a mismatch must be explained, never silent, never mysterious."""
import json
from pathlib import Path

import numpy as np

from vid2smplx.provenance import (PROVENANCE_KEY, capture_environment, code_differences,
                                  compare_environment, describe, golden_unusable,
                                  mismatch_message, read_provenance, verdict, write_golden)

ENV_A = {"gpu": "NVIDIA RTX A5000", "torch": "2.3.0", "cuda": "12.1",
         "git_commit": "a" * 40, "git_dirty": False, "git_submodules": "abc GVHMR"}
ENV_B = {**ENV_A, "gpu": "NVIDIA A100-SXM4-40GB"}


def test_same_environment_has_no_diffs():
    assert compare_environment(ENV_A, dict(ENV_A)) == []


def test_gpu_difference_is_reported():
    diffs = compare_environment(ENV_A, ENV_B)
    assert len(diffs) == 1 and "gpu" in diffs[0]
    assert "A5000" in diffs[0] and "A100" in diffs[0]


def test_torch_and_cuda_differences_are_reported():
    other = {**ENV_A, "torch": "2.1.0", "cuda": "11.8"}
    diffs = compare_environment(ENV_A, other)
    assert {d.split(":")[0] for d in diffs} == {"torch", "cuda"}


def test_a_golden_without_provenance_is_rejected_outright():
    """It must not be merely 'a mismatch' — that verdict can only skip, never fail.

    Before this, a real 0.05 rad regression (2.5x the 0.02 tight tolerance) against the
    provenance-less golden came out as `skip`: the regression test could not fail at all.
    """
    why = golden_unusable(None)
    assert why, "a golden of unknown origin must not be usable"
    assert "--update-golden" in why                      # actionable
    # and the old path really was toothless: full env mismatch + inside the wide envelope
    assert verdict(compare_environment(None, ENV_A), ["body_pose: 0.05 > 0.02"], []) == "skip"
    assert golden_unusable(ENV_A) == ""                   # a stamped golden is fine


def test_missing_provenance_counts_as_mismatch():
    """A golden that predates provenance is of unknown origin — it must not get the tight path."""
    diffs = compare_environment(None, ENV_A)
    assert len(diffs) == 3
    assert all("predates provenance" in d for d in diffs)


def test_code_differences_flag_commit_dirty_and_submodules():
    other = {**ENV_A, "git_commit": "b" * 40, "git_dirty": True, "git_submodules": "def GVHMR"}
    out = " ".join(code_differences(ENV_A, other))
    assert "git_commit" in out and "git_dirty" in out and "git_submodules" in out
    assert code_differences(ENV_A, dict(ENV_A)) == []


def test_code_difference_does_not_relax_tolerance():
    """Same machine, different commit: still the tight path — that is what the test is for."""
    other = {**ENV_A, "git_commit": "b" * 40}
    assert compare_environment(ENV_A, other) == []
    assert code_differences(ENV_A, other)


def test_message_leads_with_the_environment_then_the_numbers():
    msg = mismatch_message(ENV_A, ENV_B, compare_environment(ENV_A, ENV_B),
                           ["betas: max abs diff 0.0880 > 0.05"], over_envelope=False)
    assert msg.index("A5000") < msg.index("betas")
    assert "0.027" in msg and "0.088" in msg           # the measured drift is quoted
    assert "--update-golden" in msg                    # actionable
    assert "inconclusive" in msg.lower()


def test_message_over_envelope_says_regression_not_hardware():
    msg = mismatch_message(ENV_A, ENV_B, compare_environment(ENV_A, ENV_B),
                           ["body_pose: max abs diff 0.5400 > 0.06"], over_envelope=True)
    assert "regression" in msg.lower()
    assert "unlikely to be explained by hardware" in msg


def test_message_handles_a_golden_with_no_provenance():
    msg = mismatch_message(None, ENV_A, compare_environment(None, ENV_A), ["betas: 0.09"],
                           over_envelope=False)
    assert "<no provenance recorded>" in msg
    assert describe(None) == "<no provenance recorded>"


def test_describe_marks_dirty_trees():
    assert "+dirty" in describe({**ENV_A, "git_dirty": True})
    assert "+dirty" not in describe(ENV_A)


def test_write_golden_round_trip(tmp_path: Path):
    """It saved is not it reloads: the payload must survive and the stamp must come back."""
    src, dst = tmp_path / "src.npz", tmp_path / "golden" / "smplx_params.npz"
    body = np.random.RandomState(0).randn(7, 63).astype(np.float32)
    np.savez(src, body_pose=body, num_frames=7)
    write_golden(src, dst, ENV_A)
    got = dict(np.load(dst, allow_pickle=True))
    assert np.array_equal(got["body_pose"], body)
    assert int(got["num_frames"]) == 7
    assert read_provenance(got) == ENV_A


def test_read_provenance_absent_or_corrupt(tmp_path: Path):
    assert read_provenance({"body_pose": np.zeros(3)}) is None
    assert read_provenance({PROVENANCE_KEY: np.str_("{not json")}) is None


def test_capture_environment_never_raises_and_is_complete():
    env = capture_environment(Path(__file__).resolve().parents[1])
    for f in ("gpu", "torch", "cuda", "git_commit", "git_dirty", "python", "platform"):
        assert f in env
    assert isinstance(env["git_dirty"], bool)
    json.dumps(env)   # must be serialisable, it goes into the npz


def test_capture_environment_outside_a_git_repo(tmp_path: Path):
    env = capture_environment(tmp_path)
    assert env["git_commit"] == "unknown" and env["git_dirty"] is False


def test_verdict_same_env_tight_tolerance_decides():
    assert verdict([], [], []) == "pass"
    assert verdict([], ["body_pose: 0.54 > 0.02"], []) == "fail"


def test_verdict_cross_env_big_regression_still_fails():
    """The un-IK'd-arms bug (0.54 rad) must fail even on a foreign machine."""
    assert verdict(["gpu: ..."], ["body_pose: 0.54"], ["body_pose: 0.54 > 0.06"]) == "fail"


def test_verdict_cross_env_small_drift_is_inconclusive_not_a_pass():
    assert verdict(["gpu: ..."], ["betas: 0.088 > 0.05"], []) == "skip"
    assert verdict(["gpu: ..."], [], []) == "skip"


GOLDEN_DIR = Path(__file__).resolve().parent / "functional" / "golden"


def test_committed_goldens_carry_their_provenance():
    """The guard that keeps test_golden from switching itself off.

    Checks EVERY committed golden, not one hardcoded path: there is one per (clip, length), and a
    new one appears whenever someone runs --update-golden with a different --clip/--seconds. A
    guard naming a single file would silently stop covering the others.

    Deliberately NOT marked `functional`: it needs no GPU, no weights and no pipeline run — it only
    reads the committed bytes — so it belongs in the default suite, where a golden that cannot
    support a regression check is actually visible.
    """
    goldens = sorted(GOLDEN_DIR.glob("*.npz"))
    assert goldens, f"no goldens at all under {GOLDEN_DIR}"
    for g in goldens:
        how = (f"  bash tests/run_tests.sh uv functional --update-golden "
               f"--clip examples/{g.stem.rsplit('_', 1)[0]}.mp4 --seconds {g.stem.rsplit('_', 1)[1][:-1]}")
        ref = np.load(g, allow_pickle=True)
        assert PROVENANCE_KEY in ref.files, (
            f"{g} has no {PROVENANCE_KEY} member — it predates provenance, so test_golden cannot\n"
            f"check a regression at the tight tolerance and would silently skip instead.\n"
            f"Fix: regenerate it on a machine with the weights and a GPU —\n{how}\n"
            f"which reruns the pipeline on the seeded clip and stamps the environment "
            f"(GPU, torch, CUDA, commit, submodules) into the npz. Commit the new file.")
        assert read_provenance(dict(ref)), (
            f"{PROVENANCE_KEY} is present in {g} but does not parse — the file was written by an "
            f"older/other tool. Regenerate it:\n{how}")
