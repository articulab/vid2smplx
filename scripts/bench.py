#!/usr/bin/env python3
"""Per-stage benchmark: wall time, peak VRAM, mean GPU utilisation.

    python scripts/bench.py examples/clip_talking.mp4 --percent 10 40 100 [--extra --final-incam]

Runs `vid2smplx run` once per --percent value on a fresh output dir, samples nvidia-smi every 0.5 s and
attributes samples to the current "==== Step" banner. Prints one table per run and a JSON dump.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STEP_RE = re.compile(r"==== Step ([\d.]+)\S*:? *([A-Za-z+ ]+?) *[(=]")


def nvsmi() -> tuple[int, int]:
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                                   "--format=csv,noheader,nounits"], text=True)
    used, util = out.strip().split(",")
    return int(used), int(util)


def run_once(video: str, percent: int, extra: list[str], out_root: Path) -> dict:
    out_dir = out_root / f"p{percent}"
    shutil.rmtree(out_dir, ignore_errors=True)
    cmd = [sys.executable, "-m", "vid2smplx.cli", "run", video, "--percent", str(percent),
           "--output-dir", str(out_dir), "--skip-doctor", *extra]
    stages: dict[str, dict] = {}
    current = ["startup"]
    stop = threading.Event()
    baseline = nvsmi()[0]

    def sampler():
        while not stop.is_set():
            used, util = nvsmi()
            st = stages.setdefault(current[0], {"peak_mb": 0, "utils": [], "t0": time.time()})
            st["peak_mb"] = max(st["peak_mb"], used - baseline)
            st["utils"].append(util)
            time.sleep(0.5)

    th = threading.Thread(target=sampler, daemon=True); th.start()
    t_start = time.time()
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    frames = None
    for line in proc.stdout:
        m = STEP_RE.search(line)
        if m:
            name = f"{m.group(1)} {m.group(2).strip()}"
            stages.get(current[0], {}).setdefault("t_end", time.time())
            current[0] = name
        if "Cutting first" in line or "[Resolution]" in line or "[TIMER]" in line:
            print("   ", line.rstrip())
    proc.wait()
    stages.get(current[0], {}).setdefault("t_end", time.time())
    stop.set(); th.join()
    total = time.time() - t_start
    npz = next(out_dir.glob("*/smplx_params.npz"), None)
    if npz:
        import numpy as np
        frames = int(np.load(npz, allow_pickle=True)["num_frames"])
    rows = []
    for name, st in stages.items():
        if "t_end" not in st or name == "startup":
            continue
        rows.append({"stage": name, "sec": round(st["t_end"] - st["t0"], 1), "peak_vram_mb": st["peak_mb"],
                     "mean_util": round(sum(st["utils"]) / max(len(st["utils"]), 1))})
    return {"percent": percent, "frames": frames, "total_sec": round(total), "returncode": proc.returncode, "stages": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--percent", type=int, nargs="+", default=[10, 40, 100])
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    ap.add_argument("--out", default=str(REPO / "output" / "bench"))
    a = ap.parse_args()
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True).strip()
    results = {"gpu": gpu, "video": a.video, "runs": []}
    for p in a.percent:
        print(f"\n=== {a.video} @ {p}% ===")
        r = run_once(a.video, p, a.extra, Path(a.out))
        results["runs"].append(r)
        print(f"  frames={r['frames']} total={r['total_sec']}s rc={r['returncode']}")
        print(f"  {'stage':22s}{'sec':>8s}{'peakVRAM':>10s}{'util%':>7s}")
        for s in r["stages"]:
            print(f"  {s['stage']:22s}{s['sec']:8.1f}{s['peak_vram_mb']:8d}MB{s['mean_util']:6d}%")
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "bench.json").write_text(json.dumps(results, indent=1))
    print(f"\nGPU: {gpu}\nJSON: {Path(a.out) / 'bench.json'}")


if __name__ == "__main__":
    main()
