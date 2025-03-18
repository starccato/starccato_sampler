import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.random import PRNGKey
from jaxtyping import Array
from numpyro.infer import MCMC, NUTS

RNG = PRNGKey(0)

def log_prior(theta):
    """
    Compute the log-prior of the latent variables.
    """
    return dist.Normal(0, 1).log_prob(theta).sum()

def _bayesian_model(d: int, v: float, beta: float, reference_prior: Array = None):
    """
    Bayesian model with tempering.
    """
    # Define priors for the latent variables
    theta = numpyro.sample("z", dist.Normal(0, 1).expand([d]))

    # Compute log-likelihood (assuming a Gaussian likelihood model)
    # Use jnp.sum(theta**2) if theta is one-dimensional.
    lnl = - jnp.sum(theta ** 2) / (2 * v)
    numpyro.deterministic("untempered_loglike", lnl)

    if reference_prior is not None:
        # Calculate log-probabilities from each reference prior distribution.
        ln_ref_pri = jnp.sum(jnp.array([reference_prior[i].log_prob(theta[i]) for i in range(d)]))
        ln_pri = log_prior(theta)
        lnl = lnl + ln_pri - ln_ref_pri

    # Temper the likelihood (the power likelihood)
    numpyro.factor("likelihood", beta * lnl)

def sample_lartillot(d, v, reference_prior=None, beta=1.0):
    """
    Run MCMC sampling.
    """
    # Wrap the model function so it can be called without arguments.
    def model():
        return _bayesian_model(d=d, v=v, beta=beta, reference_prior=reference_prior)

    nuts_kernel = NUTS(model)
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=500,
        num_samples=1000,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(RNG)
    return mcmc

# Example usage:
if __name__ == '__main__':
    # Suppose we want a 3-dimensional latent space, v=2.0, and no reference_prior.
    mcmc_results = sample_lartillot(d=3, v=2.0, reference_prior=None, beta=1.0)
    print(mcmc_results.get_samples())
