import os
from time import process_time
from typing import Dict

import arviz as az
import jax.numpy as jnp
import jax.random as random
import numpy as np
from numpyro.infer import MCMC, NUTS
from starccato_jax import StarccatoVAE
from starccato_jax.credible_intervals import coverage_probability, pointwise_ci
from starccato_jax.starccato_model import StarccatoModel
from tqdm.auto import tqdm


from .core import _run_mcmc, _run_ns
from .post_proc import _post_process


def sample(
    data: jnp.ndarray,
    starccato_model: StarccatoModel = None,
    rng_int: int = 0,
    outdir="out_sampler",
    num_warmup=500,
    num_samples=1000,
    num_chains=1,
    noise_sigma=1.0,
    ns_lnz: bool = False,
    stepping_stone_lnz: bool = False,
    gss_lnz: bool = False,
    verbose=True,
    truth=None,
    **lnz_kwargs,
) -> MCMC:
    """
    Sample latent variables given the data.
    """
    rng = random.PRNGKey(rng_int)

    if starccato_model is None:
        starccato_model = StarccatoVAE()

    sampler_kwgs = dict(
        y_obs=data,
        starccato_model=starccato_model,
        num_samples=num_samples,
        num_warmup=num_warmup,
        rng=rng,
        noise_sigma=noise_sigma,
    )

    t0 = process_time()

    if ns_lnz:
        ns_obj = _run_ns(**sampler_kwgs)
        #FIXME: this is not the right way to set up the posterior for az... (Dimensions:             (chain: 5000, draw: 5000))
        inf_object = az.from_dict(
            posterior=ns_obj.get_samples(rng, num_samples),
            sample_stats=dict(ESS=int(ns_obj._results.ESS), num_samples=ns_obj._results.total_num_samples, num_lnl_evals=ns_obj._results.total_num_likelihood_evaluations),
        )
    else:
        mcmc_obj = _run_mcmc(
            **sampler_kwgs, beta=1.0, num_chains=num_chains, progress_bar=verbose
        )
        inf_object = az.from_numpyro(mcmc_obj)
    inf_object.sample_stats["sampling_runtime"] = process_time() - t0

    _post_process(
        inf_object,
        data,
        truth,
        starccato_model,
        sampler_kwgs,
        outdir,
        stepping_stone_lnz,
        gss_lnz,
        **lnz_kwargs,
    )
    return inf_object


