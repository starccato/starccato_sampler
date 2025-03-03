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


def _post_process(
    inf_object,
    data,
    truth,
    vae,
    mcmc_kwgs,
    outdir,
    verbose,
    stepping_stone_lnz,
    **lnz_kwargs,
):
    """Handles post-processing of the inference object."""
    inf_object.sample_stats["data"] = np.array(data)

    _add_quantiles(inf_object, vae)

    if truth is not None:
        _add_truth(inf_object, truth)
        _add_coverage(inf_object, vae)

    # add evidences
    _add_gaussian_approx_lnz(inf_object, **lnz_kwargs)
    if stepping_stone_lnz:
        _add_stepping_stone_lnz(mcmc_kwgs, outdir, inf_object, **lnz_kwargs)

    os.makedirs(outdir, exist_ok=True)
    _save_plots(inf_object, outdir, vae)
    inf_object.to_netcdf(os.path.join(outdir, "inference.nc"))


def _add_quantiles(
    inf_object: az.InferenceData, vae: StarccatoVAE, nsamps: int = 100
):
    zpost = _get_zposterior(inf_object, nsamps)
    posterior_predictive = vae.generate(z=zpost)
    qtls = pointwise_ci(posterior_predictive, ci=0.9)
    inf_object.sample_stats["quantiles"] = (("quantile", "samples"), qtls)


def _save_plots(inf_object: az.InferenceData, outdir: str, vae):
    data = inf_object.sample_stats["data"]
    truth = inf_object.sample_stats.get("true_signal", None)

    if truth is not None:
        truth = jnp.array(truth)

    zpost = _get_zposterior(inf_object)
    sampler_diagnostic_plots(inf_object, outdir)
    plot_ci(
        data,
        z_posterior=zpost,
        starccato_vae=vae,
        y_true=truth,
        fname=os.path.join(outdir, "ci_plot.png"),
    )


def _add_truth(inf_object: az.InferenceData, truth: jnp.ndarray):
    true_latent = None
    if isinstance(truth, dict):
        true_latent = truth["latent"]
        truth = truth["signal"]

    assert len(truth) == len(
        inf_object.sample_stats["data"]
    ), "Truth and data must have the same length."
    assert (
        truth.dtype == inf_object.sample_stats["data"].dtype
    ), "Truth and data must have the same dtype."

    inf_object.sample_stats["true_signal"] = truth
    if true_latent is not None:
        inf_object.sample_stats["true_latent"] = true_latent


def _add_coverage(inf_object: az.InferenceData, vae: StarccatoVAE):
    truth = jnp.array(inf_object.sample_stats["true_signal"])
    reconstruction_coverage = vae.reconstruction_coverage(truth, n=200, ci=0.9)
    posterior_coverage = _compute_posterior_coverage(
        truth, inf_object, 200, vae
    )
    print(f"Reconstruction coverage: {reconstruction_coverage:.2f}")
    print(f"Posterior coverage: {posterior_coverage:.2f}")

    inf_object.sample_stats[
        "reconstruction_coverage"
    ] = reconstruction_coverage
    inf_object.sample_stats["posterior_coverage"] = posterior_coverage


def _add_gaussian_approx_lnz(inf_object: az.InferenceData, **lnz_kwargs: Dict):
    """Adds Gaussian Approx LnZ to the inference object."""
    zpost = inf_object.posterior["z"].stack(sample=("chain", "draw")).values.T
    lnl = (
        inf_object.log_likelihood.stack(sample=("chain", "draw"))
        .to_array()
        .values.T
    )

    assert zpost.shape[0] == lnl.shape[0]
    map_idx = np.argmax(lnl)
    t0 = process_time()
    lnz, lnz_uncertainty = compute_gaussian_approx_evidence(
        lnl_ref=lnl[map_idx].ravel(),
        ref_sample=zpost[map_idx].ravel(),
        posterior_samples=zpost,
        n_bootstraps=lnz_kwargs.get("n_bootstraps", 10),
    )
    t1 = process_time()
    runtime = t1 - t0
    print(
        f"GA Log Z: {lnz:.3e} +/- {lnz_uncertainty:.3e} [Runtime: {runtime:.3f} s]"
    )
    inf_object.sample_stats["gauss_lnz"] = lnz
    inf_object.sample_stats["gauss_lnz_uncertainty"] = lnz_uncertainty
    inf_object.sample_stats["gauss_runtime"] = runtime


def _compute_posterior_coverage(
    data: jnp.ndarray,
    inf_object: az.InferenceData,
    num_samples: int,
    starccato_vae: StarccatoVAE,
    ci: float = 0.9,
) -> float:
    """Compute the posterior coverage."""
    zpost = _get_zposterior(inf_object)
    recon_data = starccato_vae.generate(z=zpost)
    qtls = pointwise_ci(recon_data, ci=ci)
    return coverage_probability(quantiles=qtls, true_signal=data)


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


def _get_zposterior(inf_object, nsamps: int = 100):
    zpost = inf_object.posterior["z"].stack(sample=("chain", "draw")).values.T
    z_samples = zpost[np.random.choice(zpost.shape[0], nsamps, replace=False)]
    return z_samples
