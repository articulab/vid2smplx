import pytest


def pytest_addoption(parser):
    parser.addoption("--update-golden", action="store_true",
                     help="overwrite tests/functional/golden/ with this run's smplx_params.npz")
    parser.addoption("--seconds", type=float, default=1.5, help="length of the test clip")
    parser.addoption("--clip", default="examples/clip_talking.mp4", help="source video for the test clip")
