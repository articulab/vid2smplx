#!/bin/bash
# Compatibility shim: the pipeline is now `vid2smplx run`. Old flags (--production, --final_incam, ...) still work.
if command -v vid2smplx >/dev/null; then
    exec vid2smplx run "$@"
fi
exec conda run -n "${CONDA_ENV:-vid2smplx}" --no-capture-output vid2smplx run "$@"
