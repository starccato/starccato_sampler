from time import process_time

import arviz as az
import jax.numpy as jnp
import jax.random as random
from starccato_jax import StarccatoVAE
from starccato_jax.starccato_model import StarccatoModel

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
) -> az.InferenceData:
    """
    Sample latent variables given the data.
    """
    rng = random.PRNGKey(rng_int)

    if starccato_model is None:
        starccato_model = StarccatoVAE()
    elif isinstance(starccato_model, str):
        starccato_model = StarccatoVAE(starccato_model)

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
        inf_object = _run_ns(**sampler_kwgs)

    else:
        inf_object = _run_mcmc(
            **sampler_kwgs,
            beta=1.0,
            num_chains=num_chains,
            progress_bar=verbose,
        )

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
