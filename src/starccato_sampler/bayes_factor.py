import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
from starccato_jax.plotting import add_quantiles
from starccato_jax.plotting.utils import TIME

from .sampler import sample


def compute_log_bf(
    data: jnp.ndarray,
    sampler_kwargs: dict = None,
    outdir="out_bf",
    save_plots: bool = True,
) -> float:
    """
    Compute the log Bayes factor between two models.
    """
    if sampler_kwargs is None:
        sampler_kwargs = dict(
            num_samples=1000,
            num_warmup=2000,
            num_chains=1,
            noise_sigma=1.0,
            ns_lnz=True,
            stepping_stone_lnz=False,
            gss_lnz=False,
            rng_int=0,
            save_plots=save_plots,
        )

    # Sample the data
    signal_results = sample(
        data,
        starccato_model="default_ccsne",
        outdir=f"{outdir}/signal",
        **sampler_kwargs,
    )
    blip_results = sample(
        data,
        starccato_model="default_blip",
        outdir=f"{outdir}/blip",
        **sampler_kwargs,
    )

    # Compute the log Bayes factor
    log_bf = float(
        signal_results.sample_stats.ns_lnz - blip_results.sample_stats.ns_lnz
    )

    # plot the quantiles
    _plot(
        data=data,
        signal_quantiles=signal_results.sample_stats["quantiles"],
        blip_quantiles=blip_results.sample_stats["quantiles"],
        log_bf=log_bf,
        outdir=outdir,
    )

    return log_bf


def _plot(
    data: jnp.ndarray,
    signal_quantiles: jnp.ndarray,
    blip_quantiles: jnp.ndarray,
    log_bf: float,
    outdir: str,
    time: jnp.ndarray = TIME,
):
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.plot(time, data, label="Data", color="gray", alpha=0.5)
    add_quantiles(ax, signal_quantiles, label="Signal")
    add_quantiles(ax, blip_quantiles, label="Glitch", color="tab:blue")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.legend(frameon=False)
    ax.set_title(f"Log Bayes Factor: {log_bf:.2f}")
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "quantiles.png"))
