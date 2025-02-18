from typing import Tuple

from jax import numpy as jnp
from jax.random import PRNGKey


def compute_gss_evidence(
    ln_likes: jnp.ndarray, betas: jnp.ndarray, outdir: str, rng: PRNGKey
) -> Tuple[float, float]:
    pass
