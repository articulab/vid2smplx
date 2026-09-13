"""Regression guards for four performance fixes.

These assert BEHAVIOUR, not wording. The failure mode this repo keeps hitting is a
test that checks a message was printed while the thing it names quietly stopped
happening (`--no-hands` shipped broken through three audits that way).
"""
import ast
import re
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
AUTO_BATCH = REPO / "GVHMR" / "hmr4d" / "utils" / "auto_batch.py"
VITPOSE = REPO / "GVHMR" / "hmr4d" / "utils" / "preproc" / "vitpose.py"
SLURM = REPO / "scripts" / "slurm_interp.sh"


def _load_auto_batch():
    """Import auto_batch.py from THIS checkout, not whichever hmr4d is installed."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_v2s_auto_batch", AUTO_BATCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fn_source(path, name):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(path.read_text(), node)
    raise AssertionError(f"{name} not found in {path}")


# ---------------------------------------------------------------------------
# 1. Auto batch size must respond to FREE VRAM, not total
# ---------------------------------------------------------------------------

class _FakeCuda:
    """Minimal torch.cuda stand-in: a card of `total` bytes with `free` available."""

    def __init__(self, free, total, per_sample=100e6, fixed=1e9):
        self.free, self.total = free, total
        self.per_sample, self.fixed = per_sample, fixed
        self._peak = 0

    def empty_cache(self):
        pass

    def reset_peak_memory_stats(self, device=None):
        self._peak = 0

    def max_memory_allocated(self, device=None):
        return self._peak

    def current_device(self):
        return 0

    def mem_get_info(self, device=None):
        # The real API rejects an indexless device; a lax stub hid exactly that crash.
        if not isinstance(device, int):
            raise ValueError(f"Expected an integer device index, but got: {device}")
        return (self.free, self.total)

    def note_batch(self, n):
        self._peak = self.fixed + n * self.per_sample


@pytest.fixture
def ab(monkeypatch):
    mod = _load_auto_batch()

    def run(free, total=48e9, device=0, **kw):
        fake = _FakeCuda(free, total)
        monkeypatch.setattr(mod.torch, "cuda", fake, raising=False)
        return mod.auto_batch_size(
            fake.note_batch, label="test", cap=512, device=device, **kw,
        )
    return run


def test_batch_size_tracks_free_memory(ab):
    """Same card, more free VRAM -> strictly bigger batch."""
    assert ab(free=40e9) > ab(free=10e9) > ab(free=3e9)


def test_cotenant_process_shrinks_the_batch(ab):
    """The measured bug: 3 pipelines on one 48GB card each claimed the whole card.

    A second pipeline holding 2/3 of the card must shrink our batch, not be ignored.
    """
    alone = ab(free=46e9, total=48e9)
    crowded = ab(free=15e9, total=48e9)
    assert crowded < alone, (
        f"batch did not shrink with a co-tenant on the card: {crowded} vs {alone}"
    )


def test_total_memory_does_not_drive_the_batch(ab):
    """Two cards of very different SIZE but identical FREE bytes agree.

    This is what fails if anyone reintroduces get_device_properties().total_memory.
    """
    big = ab(free=8e9, total=48e9)
    small = ab(free=8e9, total=24e9)
    assert big == small


def test_safety_factor_and_cap_are_respected(ab):
    assert ab(free=40e9, safety_factor=4) < ab(free=40e9, safety_factor=1)
    assert ab(free=1e12) <= 512


def test_batch_is_never_zero(ab):
    """A nearly-full card must still return a runnable batch, not 0."""
    assert ab(free=1e6) >= 1


def test_small_card_is_clamped_harder(ab):
    """The <12GB safety must survive consolidation."""
    assert ab(free=5e9, total=6e9, small_gpu_divisor=4) < ab(free=5e9, total=48e9)


def test_small_card_target_util_override(ab):
    """HaMeR's extra 0.65 util on small cards must not have been dropped."""
    assert ab(free=5e9, total=6e9, small_gpu_target_util=0.65) <= ab(free=5e9, total=6e9)


@pytest.mark.parametrize("device", [None, "cuda", "cuda:0", torch.device("cuda"), 0])
def test_accepts_the_device_forms_the_call_sites_actually_pass(ab, device):
    """EMICA and HaMeR pass a bare torch.device('cuda'); mem_get_info rejects that.

    Caught on real hardware, not by a stub -- hence the strict stub above.
    """
    assert ab(free=20e9, device=device) >= 1


BATCH_SIZE_SITES = [
    (AUTO_BATCH, None),
    (REPO / "scripts" / "utils.py", "auto_emica_batch_size"),
    (REPO / "hamer" / "demo.py", "auto_find_batch_size"),
    (VITPOSE, "extract"),
    (REPO / "GVHMR" / "hmr4d" / "utils" / "preproc" / "vitfeat_extractor.py",
     "extract_video_features"),
]


@pytest.mark.parametrize("path,fn", BATCH_SIZE_SITES, ids=lambda v: getattr(v, "name", str(v)))
def test_no_total_memory_in_any_batch_size_path(path, fn):
    """total_memory must not come back in any batch-sizing code."""
    src = path.read_text() if fn is None else _fn_source(path, fn)
    src = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    assert "get_device_properties" not in src, f"{path.name}:{fn} sizes against total memory again"


def test_there_is_exactly_one_copy_of_the_heuristic():
    """Four near-duplicate copies is how they all shared one bug. Keep it at one."""
    owners = [p for p, _ in BATCH_SIZE_SITES[1:]
              if re.search(r"peak4\s*-\s*peak1", p.read_text())]
    assert owners == [], f"a second copy of the two-probe heuristic reappeared in {owners}"
    assert re.search(r"peak4\s*-\s*peak1", AUTO_BATCH.read_text())


# ---------------------------------------------------------------------------
# 2. seed_everything seeds RNGs and nothing else
# ---------------------------------------------------------------------------

def _seed_everything():
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        from utils import seed_everything
    finally:
        sys.path.pop(0)
    return seed_everything


def test_seed_everything_does_not_touch_cudnn_flags():
    """These are process-wide perf flags, not RNG state; they cost up to 2.7x."""
    seed_everything = _seed_everything()
    before = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    flipped = (not before[0], not before[1])
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = flipped
    try:
        seed_everything(1234)
        after = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
        assert after == flipped, f"seed_everything mutated cudnn flags: {flipped} -> {after}"
    finally:
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = before


def test_seed_everything_still_seeds_every_rng():
    """Scoping it down must not have scoped away the actual job."""
    import random
    import numpy as np
    seed_everything = _seed_everything()

    seed_everything(7)
    first = (random.random(), np.random.rand(), torch.rand(1).item())
    seed_everything(7)
    assert (random.random(), np.random.rand(), torch.rand(1).item()) == first

    seed_everything(8)
    assert (random.random(), np.random.rand(), torch.rand(1).item()) != first


# ---------------------------------------------------------------------------
# 3. ViTPose runs under fp16 autocast
# ---------------------------------------------------------------------------

def test_vitpose_forward_applies_fp16_autocast(monkeypatch):
    """Behavioural: the stub reports the autocast state it was actually called under."""
    from hmr4d.utils.preproc.vitpose import VitPoseExtractor
    monkeypatch.delenv("VID2SMPLX_VITPOSE_FP16", raising=False)

    seen = {}

    def _autocast_state():
        # torch renamed these in 2.4; support both so the guard cannot rot into a skip.
        try:
            return torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda")
        except TypeError:
            return torch.is_autocast_enabled(), torch.get_autocast_gpu_dtype()

    def fake_pose(x):
        seen["enabled"], seen["dtype"] = _autocast_state()
        return torch.zeros(x.shape[0], 17, 64, 48)

    ex = VitPoseExtractor.__new__(VitPoseExtractor)   # skip __init__: it loads 2.5GB
    ex.pose = fake_pose
    out = ex._forward(torch.zeros(2, 3, 256, 192))

    assert seen["enabled"], "ViTPose forward ran with autocast OFF"
    assert seen["dtype"] is torch.float16, (
        f"expected fp16; bf16 measured 0.67x (SLOWER) on sm_7.5, got {seen['dtype']}"
    )
    assert out.dtype is torch.float32, "heatmaps must leave the forward as fp32"


def test_every_vitpose_inference_goes_through_the_autocast_wrapper():
    """A bare self.pose(...) added later would silently opt back out of fp16."""
    src = VITPOSE.read_text()
    tree = ast.parse(src)
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name in ("_forward", "__init__"):
            continue
        for call in ast.walk(node):
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "pose"
                    and isinstance(call.func.value, ast.Name) and call.func.value.id == "self"):
                offenders.append(f"{node.name}:{call.lineno}")
    assert offenders == [], f"self.pose() called outside _forward (bypasses fp16): {offenders}"


def test_fp32_escape_hatch_actually_disables_autocast(monkeypatch):
    """The A/B control must really be a control, not a no-op flag."""
    from hmr4d.utils.preproc.vitpose import VitPoseExtractor
    monkeypatch.setenv("VID2SMPLX_VITPOSE_FP16", "0")
    seen = {}

    def fake_pose(x):
        try:
            seen["enabled"] = torch.is_autocast_enabled("cuda")
        except TypeError:
            seen["enabled"] = torch.is_autocast_enabled()
        return torch.zeros(x.shape[0], 17, 64, 48)

    ex = VitPoseExtractor.__new__(VitPoseExtractor)
    ex.pose = fake_pose
    ex._forward(torch.zeros(1, 3, 256, 192))
    assert seen["enabled"] is False, "VID2SMPLX_VITPOSE_FP16=0 did not disable autocast"


# ---------------------------------------------------------------------------
# 4. SLURM array concurrency
# ---------------------------------------------------------------------------

def _sbatch(flag):
    m = re.search(rf"^#SBATCH --{flag}=(\S+)", SLURM.read_text(), re.M)
    assert m, f"no #SBATCH --{flag} directive in {SLURM.name}"
    return m.group(1)


def test_array_is_not_serialised():
    """%1 made the corpus pay 96 sequential queue waits."""
    spec = _sbatch("array")
    m = re.search(r"%(\d+)$", spec)
    assert m, f"array spec {spec!r} sets no concurrency limit"
    n = int(m.group(1))
    assert n > 1, f"array is still serialised at %{n}"
    assert n <= 8, f"%{n} exceeds the QOS `normal` cap of 8 concurrent GPUs"


def test_concurrency_is_overridable_without_editing_the_file():
    """The submit line wins over a directive; that escape hatch must be documented."""
    text = SLURM.read_text()
    assert re.search(r"sbatch .*--array=\S*%\S", text), (
        "no documented sbatch --array override; concurrency is effectively baked in"
    )


def test_walltime_lets_the_backfill_scheduler_slot_tasks():
    """24h reservations do not backfill; the median clip is 1.4h."""
    hours = int(_sbatch("time").split(":")[0])
    assert hours <= 12, f"--time={hours}h is too long to backfill"
    assert hours >= 3, f"--time={hours}h leaves no margin over a 1.4h median clip"


def test_overrun_is_recoverable():
    """A shorter walltime is only safe because a killed task resumes."""
    text = SLURM.read_text()
    assert "#SBATCH --requeue" in text
