import os

import arviz as az
import jax.numpy as jnp
import jax.random as random
from numpyro.infer import MCMC, NUTS
from starccato_jax import StarccatoVAE
from tqdm.auto import tqdm

from .core import _run_mcmc
from .evidence import compute_stepping_stone_evidence
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
    num_temps=1,
    verbose=True,
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
    kwgs = dict(
        y_obs=data,
        starccato_vae=starccato_vae,
        num_samples=num_samples,
        num_warmup=num_warmup,
        rng=rng,
    )

    mcmc = _run_mcmc(
        **kwgs, beta=1.0, num_chains=num_chains, progress_bar=verbose
    )
    inf_object = az.from_numpyro(mcmc)

    if num_temps > 3:
        tempering_betas = beta_spaced_samples(
            num_temps,
            0.3,
            1,
        )
        tempered_lnls = []
        for i in tqdm(range(num_temps), "Tempering"):
            temp_mcmc = _run_mcmc(
                **kwgs,
                beta=tempering_betas[i],
                num_chains=1,
                progress_bar=False,
            )
            tempered_lnls.append(temp_mcmc.get_samples()["untempered_loglike"])

        tempered_lnls = jnp.array(tempered_lnls).T
        log_z, log_z_uncertainty = compute_stepping_stone_evidence(
            tempered_lnls, tempering_betas, outdir, rng
        )
        print(f"Log Z: {log_z:.3e} +/- {log_z_uncertainty:.3e}")
        # add these to the inference object
        inf_object.sample_stats["log_z"] = log_z
        inf_object.sample_stats["log_z_uncertainty"] = log_z_uncertainty

    os.makedirs(outdir, exist_ok=True)

    if verbose:
        print(az.summary(inf_object, var_names=["z"]))
        sampler_diagnostic_plots(inf_object, outdir)
    z_samples = inf_object.posterior["z"].values.reshape(
        -1, starccato_vae.latent_dim
    )
    plot_ci(
        data,
        z_samples,
        starccato_vae,
        fname=os.path.join(outdir, "ci_plot.png"),
    )

    inf_object.to_netcdf(os.path.join(outdir, "inference.nc"))

    return mcmc
