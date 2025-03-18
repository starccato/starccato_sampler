import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from jax.random import PRNGKey
from numpyro.infer import MCMC, NUTS
from starccato_jax.starccato_model import StarccatoModel
from jaxtyping import Array

def log_prior(theta):
    """
    Compute the log-prior of the latent variables.

    Parameters:
      theta: latent variables.
    """
    return dist.Normal(0, 1).log_prob(theta).sum()



def log_likelihood(
        y_obs,
        y_model,
        noise_sigma=1.0,
):
    """
    Compute the log-likelihood of the data given the model.

    Parameters:
      y_obs: observed data.
      y_model: model data.
    """
    return -0.5 * jnp.sum((y_obs - y_model) ** 2) / noise_sigma ** 2



def _bayesian_model(
    y_obs: jnp.ndarray,
    starccato_model: StarccatoModel,
    rng: PRNGKey,
    beta: float,
    noise_sigma: float = 1.0,
    reference_prior: Array = None
):
    """
    Bayesian model with tempering.

    Parameters:
      y_obs   : observed data.
      vae_data: model data (e.g. containing latent_dim).
      beta    : tempering parameter; beta=1 corresponds to full posterior.
    """

    dims = starccato_model.latent_dim

    # Define priors for the latent variables
    theta = numpyro.sample(
        "z", dist.Normal(0, 1).expand([dims])
    )
    # Generate the signal
    y_model = starccato_model.generate(z=theta, rng=rng)

    # sigma = numpyro.sample("sigma", dist.HalfNormal(1))  # Noise level

    # Compute the untempered log–likelihood (Assuming Gaussian noise)
    lnl = log_likelihood(y_obs, y_model, noise_sigma=noise_sigma)
    # Save the untempered log–likelihood (for LnZ computation).
    numpyro.deterministic("untempered_loglike", lnl)

    if reference_prior is not None:
        ln_ref_pri = jnp.sum(jnp.array([reference_prior[i].log_prob(theta[i]) for i in range(dims)]))
        ln_pri = log_prior(theta)
        lnl = lnl + ln_pri - ln_ref_pri

    # Temper the likelihood (the power likelihood)
    numpyro.factor("likelihood", beta * lnl)



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
) -> MCMC:
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
    return mcmc
