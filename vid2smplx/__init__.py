"""vid2smplx: video -> SMPL-X parameters."""

# The one list of renderable layers. scripts/render.py validates against it and the
# CLI offers it as argparse choices -- defined here because this module imports nothing.
VALID_LAYERS = ("gvhmr", "hands", "face", "final", "global")
