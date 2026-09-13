"""GVHMR's banded attention vs the dense (L, L) mask it replaced.

Needs no weights, no video and no GPU (it falls back to CPU), runs in ~4 s — so it belongs
in the default suite. It lived under `functional` and was therefore never run, while being
the change with the widest blast radius in the repo: a wrong attention window still produces
plausible motion.
"""

def test_banded_attention_matches_dense_mask():
    """GVHMR's banded attention must equal the dense (L, L) mask it replaced.

    The dense mask costs a 38.10 GiB score tensor at 35,755 frames, so long videos now take
    a blocked path that never materialises it. That path is only safe while it attends to
    EXACTLY the same window, and a wrong window would still produce plausible motion — so
    this compares the two implementations directly, with and without padding.
    """
    import torch
    from hmr4d.network.base_arch.transformer.encoder_rope import BandMask, RoPEAttention

    def dense(lo, hi, L):
        j = torch.arange(L, device=lo.device)
        return ~((j[None] >= lo[:, None]) & (j[None] < hi[:, None]))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    att = RoPEAttention(512, 8, dropout=0.1).to(dev).eval()
    for L, valid in [(121, 121), (500, 463), (3000, 3000)]:
        h, ml = 60, 120
        i = torch.arange(L, device=dev)
        lo = torch.clamp(i - h, min=0).clamp(max=L - ml)
        hi = torch.clamp(i + h, max=L).clamp(min=ml)
        x = torch.randn(2, L, 512, device=dev)
        kp = torch.zeros(2, L, dtype=torch.bool, device=dev)
        kp[:, valid:] = True
        with torch.no_grad():
            a = att(x, attn_mask=dense(lo, hi, L), key_padding_mask=kp)
            b = att(x, attn_mask=BandMask(lo, hi), key_padding_mask=kp)
        assert not torch.isnan(b).any(), f"banded attention produced NaN at L={L}"
        rel = ((a - b).abs().max() / a.abs().max()).item()
        assert rel < 1e-5, f"L={L}: banded vs dense relative error {rel:.3e} — not just fp32 rounding"
