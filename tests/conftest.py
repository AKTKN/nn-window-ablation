"""Initialize numerical limits before test collection in each worker."""

import os

import pytest
from hydra import compose, initialize_config_module
from xdist import is_xdist_controller

from ._cpu import configure_torch_threads


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session: pytest.Session) -> None:
    if is_xdist_controller(session):
        return

    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"

    configure_torch_threads()


@pytest.fixture
def compose_settings():
    """Compose the installed defaults with an explicit, reproducible run name."""
    with initialize_config_module(config_module="nn_window_ablation.conf", version_base="1.3"):
        def settings(*overrides: str):
            return compose(config_name="config", overrides=["run_name=p0-unit", *overrides])

        yield settings
