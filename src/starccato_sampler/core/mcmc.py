
from numpyro.infer import MCMC, NUTS
import jax.numpy as jnp
from jax.random import PRNGKey

from starccato_jax.starccato_model import StarccatoModel
from jaxtyping import Array
from .bayesian_functions import _bayesian_model
import arviz as az


def _run_mcmc(
    y_obs: jnp.ndarray,
    starccato_model: StarccatoModel,
    num_samples: int,
    num_warmup: int,
    num_chains: int,
    rng: PRNGKey,
    beta: float = 1.0,
    progress_bar: bool = False,
    noise_sigma: float = 1.0,
    reference_prior: Array = None
) -> az.InferenceData:
    """
    Run MCMC sampling.

    Parameters:
      y_obs       : observed data.
      vae_data    : model data (e.g. containing latent_dim).
      num_samples : number of samples to draw.
      num_warmup  : number of warmup steps.
      num_chains  : number of chains.
      rng         : random number generator.
      beta        : tempering parameter; beta=1 corresponds to full posterior.
    """
    nuts_kernel = NUTS(
        lambda y_obs: _bayesian_model(
            y_obs, starccato_model, rng, beta=beta, noise_sigma=noise_sigma, reference_prior=reference_prior
        )
    )
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )
    mcmc.run(rng, y_obs=y_obs)
    return az.from_numpyro(mcmc)
