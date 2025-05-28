import os
import subprocess

import jax
import pytest
from starccato_jax import StarccatoVAE


@pytest.fixture
def outdir() -> str:
    branch = _get_branch_name()
    dir = os.path.join(os.path.dirname(__file__), f"test_output[{branch}]")
    os.makedirs(dir, exist_ok=True)
    return dir


def _get_branch_name() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
    )
    branch = result.stdout.strip() if result.returncode == 0 else "main"
    return branch


@pytest.fixture
def injection():
    vae = StarccatoVAE()
    # true_z = jax.random.normal(RNG, (1, vae.latent_dim))
    true_z = jax.numpy.zeros((1, vae.latent_dim))
    true_signal = vae.generate(z=true_z)[0]
    return true_signal, true_z.ravel()
