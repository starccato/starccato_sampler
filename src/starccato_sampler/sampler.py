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
from tqdm.auto import tqdm

from .core import _run_mcmc
from .evidence import (
    compute_gaussian_approx_evidence,
    compute_stepping_stone_evidence,
)
from .plotting import plot_ci, sampler_diagnostic_plots
from .utils import beta_spaced_samples


def sample(
    data: jnp.ndarray,
    model_path: str = None,
    rng_int: int = 0,
    outdir="out_mcmc",
    num_warmup=500,
    num_samples=1000,
    num_chains=1,
    noise_sigma=1.0,
    stepping_stone_lnz: bool = False,
    verbose=True,
    truth=None,
    **lnz_kwargs,
) -> MCMC:
    """
    Sample latent variables given the data.

    :param data: Data to condition the latent variables on.
    :param rng: Random number generator.
    :param latent_dim: Dimension of the latent space.
    :return: Sampled latent variables.
    """

    rng = random.PRNGKey(rng_int)
    starccato_vae = StarccatoVAE(model_path)

    # unpack the truth
    true_latent = None
    if truth is not None:
        if isinstance(truth, dict):
            true_latent = truth["latent"]
            truth = truth["signal"]

        reconstruction_coverage = starccato_vae.reconstruction_coverage(
            truth, n=200, ci=0.9
        )
        print(f"Reconstruction coverage: {reconstruction_coverage:.2f}")

    mcmc_kwgs = dict(
        y_obs=data,
        starccato_vae=starccato_vae,
        num_samples=num_samples,
        num_warmup=num_warmup,
        rng=rng,
        noise_sigma=noise_sigma,
    )

    t0 = process_time()
    mcmc = _run_mcmc(
        **mcmc_kwgs, beta=1.0, num_chains=num_chains, progress_bar=verbose
    )
    inf_object = az.from_numpyro(mcmc)
    t1 = process_time()

    # save some data
    inf_object.sample_stats["sampling_runtime"] = t1 - t0
    inf_object.sample_stats["data"] = np.array(data)
    inf_object.sample_stats["true_signal"] = truth
    inf_object.sample_stats["true_latent"] = true_latent

    _add_gaussian_approx_lnz(inf_object, **lnz_kwargs)
    if stepping_stone_lnz:
        _add_stepping_stone_lnz(mcmc_kwgs, outdir, inf_object, **lnz_kwargs)

    os.makedirs(outdir, exist_ok=True)

    if verbose:
        print(az.summary(inf_object, var_names=["z"]))
        sampler_diagnostic_plots(inf_object, outdir)

    plot_ci(
        data,
        z_posterior=inf_object.posterior["z"].values.reshape(
            -1, starccato_vae.latent_dim
        ),
        starccato_vae=starccato_vae,
        y_true=truth,
        fname=os.path.join(outdir, "ci_plot.png"),
    )

    if truth is not None:
        posterior_coverage = _compute_posterior_coverage(
            truth, inf_object, 200, starccato_vae
        )
        print(f"Posterior coverage: {posterior_coverage:.2f}")
        inf_object.sample_stats[
            "reconstruction_coverage"
        ] = reconstruction_coverage
        inf_object.sample_stats["posterior_coverage"] = posterior_coverage

    inf_object.to_netcdf(os.path.join(outdir, "inference.nc"))

    return mcmc


def _add_gaussian_approx_lnz(inf_object: az.InferenceData, **lnz_kwargs: Dict):
    """Adds Gaussian Approx LnZ to the inference object."""
    zpost = inf_object.posterior["z"].stack(sample=("chain", "draw")).values.T
    lnl = (
        inf_object.log_likelihood.stack(sample=("chain", "draw"))
        .to_array()
        .values.T
    )

    # Shapes --> zpost: (num_samples, latent_dim), lnl: (num_samples,)
    assert zpost.shape[0] == lnl.shape[0]  # same number of samples
    map_idx = np.argmax(lnl)
    t0 = process_time()
    lnz, lnz_uncertainty = compute_gaussian_approx_evidence(
        lnl_ref=lnl[map_idx].ravel(),
        ref_sample=zpost[map_idx].ravel(),
        posterior_samples=zpost,
        n_bootstraps=lnz_kwargs.get("n_bootstraps", 10),
    )
    t1 = process_time()
    runtime = t1 - t0  # in seconds
    print(
        f"GA Log Z: {lnz:.3e} +/- {lnz_uncertainty:.3e} [Runtime: {runtime:.3f} s]"
    )
    # add these to the inference object
    inf_object.sample_stats["gauss_lnz"] = lnz
    inf_object.sample_stats["gauss_lnz_uncertainty"] = lnz_uncertainty
    inf_object.sample_stats["gauss_runtime"] = runtime


def _add_stepping_stone_lnz(
    mcmc_kwgs: Dict,
    outdir: str,
    inf_object: az.InferenceData,
    **lnz_kwargs: Dict,
) -> None:
    """Adds Stepping Stone LnZ to the inference object."""
    num_temps = lnz_kwargs.get("num_temps", 16)
    tempering_betas = beta_spaced_samples(
        num_temps,
        0.3,
        1,
    )
    tempered_lnls = np.zeros((mcmc_kwgs["num_samples"], num_temps))
    t0 = process_time()
    for i in tqdm(range(num_temps), "Tempering"):
        temp_mcmc = _run_mcmc(
            **mcmc_kwgs,
            beta=tempering_betas[i],
            num_chains=1,
            progress_bar=False,
        )
        tempered_lnls[:, i] = temp_mcmc.get_samples()["untempered_loglike"]

    log_z, log_z_uncertainty = compute_stepping_stone_evidence(
        tempered_lnls, tempering_betas, outdir, mcmc_kwgs["rng"]
    )
    t1 = process_time()
    runtime = t1 - t0  # in seconds

    print(
        f"SS Log Z: {log_z:.3e} +/- {log_z_uncertainty:.3e} [Runtime: {runtime:.3f} s]"
    )
    # add these to the inference object
    inf_object.sample_stats["ss_lnz"] = log_z
    inf_object.sample_stats["ss_lnz_uncertainty"] = log_z_uncertainty
    inf_object.sample_stats["ss_runtime"] = runtime


def _compute_posterior_coverage(
    data: jnp.ndarray,
    inf_object: az.InferenceData,
    num_samples: int,
    starccato_vae: StarccatoVAE,
    ci: float = 0.9,
) -> float:
    """Compute the posterior coverage."""
    zpost = inf_object.posterior["z"].stack(sample=("chain", "draw")).values.T
    # only keep a subset of the samples
    zpost = zpost[:num_samples]
    recon_data = starccato_vae.generate(z=zpost)
    qtls = pointwise_ci(recon_data, ci=ci)
    coverage = coverage_probability(quantiles=qtls, true_signal=data)
    return coverage
