import jax.numpy as jnp
import jax.random
from jax.random import PRNGKey
import numpy as np
import arviz as az
from starccato_jax.starccato_model import StarccatoModel
from jaxtyping import Array
from numpyro.contrib.nested_sampling import NestedSampler
from .bayesian_functions import _bayesian_model


def _run_ns(
        y_obs: jnp.ndarray,
        starccato_model: StarccatoModel,
        num_samples: int,
        num_warmup: int,
        rng: PRNGKey,
        num_chains: int = 1,
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
    ns = NestedSampler(
        _bayesian_model, constructor_kwargs=dict(max_samples=num_samples)
    )
    ns.run(
        rng,
        y_obs=y_obs,
        starccato_model=starccato_model,
        beta=beta,
        noise_sigma=noise_sigma,
        rng=rng,
    )
    ns.print_summary()
    return _to_inference_obj(ns, rng, num_samples, starccato_model.latent_dim)


def _to_inference_obj(ns_obj: NestedSampler, rng: jax.random.PRNGKey, num_samples: int, ndim: int):
    post = ns_obj.get_samples(rng, num_samples=num_samples)
    lnl = post["untempered_loglike"][np.newaxis, :]
    stats = dict(
        ESS=int(ns_obj._results.ESS),
        num_samples=ns_obj._results.total_num_samples,
        num_lnl_evals=ns_obj._results.total_num_likelihood_evaluations,
        ns_lnz=ns_obj._results.log_Z_mean,
        ns_lnz_uncertainty=ns_obj._results.log_Z_uncert,
    )
    return az.from_dict(
        posterior=dict(
            z=post["z"][np.newaxis, :, :],
            untempered_loglike=lnl,
        ),
        coords=dict(z_dim_0=np.arange(ndim)),
        dims=dict(z=["z_dim_0"]),
        sample_stats=stats,
        log_likelihood=dict(
            likelihood=lnl,
        )
    )
