#!/bin/bash
# Run the test suite inside the conda env or the uv .venv.
#   bash tests/run_tests.sh conda            # unit tests (fast, no GPU)
#   bash tests/run_tests.sh uv functional    # real models on a 1.5 s clip (GPU, ~5 min)
#   bash tests/run_tests.sh conda all
#   bash tests/run_tests.sh uv functional --update-golden
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:?usage: run_tests.sh <conda|uv> [unit|functional|all] [pytest args...]}"
SUITE="${2:-unit}"; shift; shift || true
case "$MODE" in
    conda) PY=(conda run -n "${CONDA_ENV:-vid2smplx}" --no-capture-output python) ;;
    uv)    PY=("$REPO/.venv/bin/python") ;;
    *) echo "mode must be conda or uv"; exit 1 ;;
esac
"${PY[@]}" -c "import pytest" 2>/dev/null || "${PY[@]}" -m pip install -q pytest matplotlib
case "$SUITE" in
    unit)       "${PY[@]}" -m pytest "$REPO/tests" -m "not functional" "$@" ;;
    functional) "${PY[@]}" -m pytest "$REPO/tests/functional" -m functional -x -s "$@" ;;
    all)        "${PY[@]}" -m pytest "$REPO/tests" -m "" "$@" ;;
esac
