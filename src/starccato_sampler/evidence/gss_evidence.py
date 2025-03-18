from jax import numpy as jnp
from jax.random import PRNGKey
from jaxtyping import Array, Float
from ..core import log_likelihood, log_prior
import numpyro.distributions as dist
from .stepping_stone_evidence import compute_stepping_stone_evidence


def generate_reference_prior(posterior_samples:Array)->Array:
    mu, sigma = posterior_samples.mean(0), posterior_samples.std(0)
    return [dist.Normal(mu, sigma) for mu, sigma in zip(mu, sigma)]
