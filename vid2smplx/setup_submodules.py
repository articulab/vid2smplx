"""Verify that the git submodule checkouts contain the changes vid2smplx depends on.

GVHMR is no longer an upstream pin: it is our OWN fork (see .gitmodules), and the
changes vid2smplx needs -- the `--person` rank, the track inventory, and persisting
that inventory into preprocess/bbx.pt -- are COMMITTED on that fork. A correct
`git submodule update --init GVHMR` therefore delivers the patched code directly;
there is nothing left to apply.

What used to live here was a second copy of those same edits as vendored patch text.
Two sources of truth for one change is a drift trap: edit the fork, and the vendored
patch silently disagrees. The patch bodies and the application logic are deleted; what
remains is a CHECK.

So this module only asks: does the checked-out submodule actually contain the marker
strings? If not, the submodule is at the wrong commit — which is an actionable fact
(run the submodule update; if that does not fix it, the pin recorded in this repo is
stale), not something to repair by rewriting files under GVHMR/.

Run: `vid2smplx setup` (install.sh calls it AFTER `git submodule update`), or
`python -m vid2smplx.setup_submodules`. `vid2smplx doctor` reports the same rows, so an
unpatched tree cannot silently pass.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GVHMR = REPO / "GVHMR"

# (submodule, submodule-relative path, human name, marker string that must be present)
# Markers are the distinctive identifiers the fork's commit introduced. They are checked,
# never written: if one is absent the checkout is at the wrong commit.
REQUIRED_MARKERS = [
    ("GVHMR", "hmr4d/utils/preproc/tracker.py", "track inventory + --person rank",
     "self.track_summary"),
    ("GVHMR", "tools/demo/demo.py", "--person CLI flag", '"--person"'),
    ("GVHMR", "tools/demo/demo.py", "person -> hydra override", "+person={args.person}"),
    ("GVHMR", "tools/demo/demo.py", "persist track_summary into bbx.pt",
     '"track_summary": tracker.track_summary'),
]

SUBMODULE_DIRS = {"GVHMR": GVHMR}


class PatchError(RuntimeError):
    """A required submodule change is absent. Actionable: names the file and what to run."""


def expected_pin(sub: str, repo: Path = REPO) -> str:
    """The commit this repo pins `sub` to, or '' if it cannot be read (not a git tree)."""
    # index, not HEAD: a pin bump that is staged but not yet committed is still the pin.
    try:
        out = subprocess.run(["git", "-C", str(repo), "ls-files", "-s", "--", sub],
                             capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return ""
    parts = out.stdout.split()
    return parts[1] if out.returncode == 0 and len(parts) >= 3 else ""


def _wrong_commit_message(sub: str, rel: str, name: str, repo: Path) -> str:
    pin = expected_pin(sub, repo)
    pin_txt = f"pinned commit {pin}" if pin else "the commit pinned by this repo"
    return (
        f"'{name}' is missing from {sub}/{rel}: the submodule is checked out at the WRONG COMMIT.\n"
        f"  vid2smplx needs {sub} at {pin_txt} (our fork, see .gitmodules), which carries this "
        f"change. Without it GVHMR reconstructs one person and reports nothing about anyone else "
        f"in frame, so a two-person clip comes back as a clean single-person SUCCESS.\n"
        f"  Fix: git -C {repo} submodule update --init {sub}\n"
        f"  If that does not fix it, the submodule pin recorded in this repo is stale: update it to "
        f"a fork commit that contains the change (git -C {repo}/{sub} fetch && git -C {repo}/{sub} "
        f"checkout origin/main, then commit the new pin)."
    )


def _submodule_path(sub: str, rel: str, repo: Path) -> Path:
    root = repo / sub
    if not root.is_dir() or not any(root.iterdir()):
        raise PatchError(
            f"[setup] submodule '{sub}' is not checked out at {root}. "
            f"Run: git -C {repo} submodule update --init --recursive {sub}")
    p = root / rel
    if not p.is_file():
        raise PatchError(
            f"[setup] {p} is missing. The '{sub}' submodule is checked out but does not contain "
            f"the file vid2smplx needs — it is at the wrong commit. "
            f"Run: git -C {repo} submodule update --init {sub}")
    return p


def verify_marker(sub: str, rel: str, name: str, marker: str, repo: Path = REPO) -> str:
    """Check one required change is present. Returns 'ok'. Raises PatchError when absent."""
    path = _submodule_path(sub, rel, repo)
    if marker not in path.read_text(encoding="utf-8"):
        raise PatchError("[setup] " + _wrong_commit_message(sub, rel, name, repo))
    return "ok"


def patch_status(repo: Path = REPO) -> list[tuple[str, str, str]]:
    """(status, label, note) per required change, WITHOUT modifying anything. For doctor.

    status is 'OK' (marker present) or 'MISS' (absent — submodule is at the wrong commit).
    """
    rows = []
    for sub, rel, name, marker in REQUIRED_MARKERS:
        label = f"patch: {sub}/{rel} ({name})"
        try:
            verify_marker(sub, rel, name, marker, repo)
        except PatchError as e:
            rows.append(("MISS", label, str(e).replace("[setup] ", "", 1)))
        else:
            rows.append(("OK", label, ""))
    return rows


def main(argv: list[str] | None = None) -> int:
    repo = REPO
    print(f"[setup] repo: {repo}")
    pin = expected_pin("GVHMR", repo)
    if pin:
        print(f"[setup] GVHMR pinned at {pin}")
    failed = 0
    for sub, rel, name, marker in REQUIRED_MARKERS:
        try:
            verify_marker(sub, rel, name, marker, repo)
        except PatchError as e:
            # full remedy once; the rest are one line each, same cause, same fix.
            print(str(e) if not failed else f"[setup] also missing: {name} ({sub}/{rel})",
                  file=sys.stderr)
            failed += 1
            continue
        print(f"[setup] {sub}/{rel}: {name} -> present")
    if failed:
        print(f"[setup] FAILED — {failed} required submodule change(s) missing.", file=sys.stderr)
        return 1
    print(f"[setup] done — all {len(REQUIRED_MARKERS)} required submodule change(s) present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
