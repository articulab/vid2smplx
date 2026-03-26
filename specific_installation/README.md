# Specific Installation Guides

The main `install.sh` targets CUDA 12.1 with sm_70–sm_90 (RTX 8000, A100, H100, H200). Some GPU architectures require separate builds because:

- **PyTorch3D** and **nvdiffrast** compile CUDA kernels at install time against a specific compute capability (`sm_XX`). A binary built for sm_90 won't run on sm_120 and vice versa.
- **PyTorch nightly** may be required for newer architectures not yet in stable releases.
- **CUDA toolkit version** must match the PyTorch build (e.g. Blackwell needs CUDA 12.8+).

## Blackwell (sm_120) — RTX PRO 1000, RTX 5090, etc.

Requires PyTorch nightly with CUDA 12.8 and `TORCH_CUDA_ARCH_LIST="12.0"` for all compiled extensions.

```bash
# Create the environment
bash specific_installation/env_blackwell.sh

# Activate
conda activate vid2smplx_bw

# Frozen deps (if env_blackwell.sh fails on version resolution)
pip install -r specific_installation/requirements_blackwell.txt
```

Key differences from main install:
- PyTorch 2.10+ nightly (CUDA 12.8)
- `TORCH_CUDA_ARCH_LIST="12.0"` for PyTorch3D and nvdiffrast compilation
- `setuptools<71` pinned (mmcv needs `pkg_resources`)
- 8GB VRAM: HaMeR uses deferred model loading (ViTPose runs first, freed, then HaMeR loads)
