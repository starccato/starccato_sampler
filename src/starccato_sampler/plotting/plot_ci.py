import os
from typing import List

import arviz as az
import corner
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from starccato_jax import StarccatoVAE
from starccato_jax.credible_intervals import coverage_probability, pointwise_ci
from starccato_jax.plotting import add_quantiles
from starccato_jax.plotting.utils import TIME
from starccato_jax.starccato_model import StarccatoModel


def plot_ci(
    y_obs: jnp.ndarray,
    z_posterior: jnp.ndarray,
    starccato_model: StarccatoModel,
    y_true: jnp.ndarray = None,
    nsamps=100,
    fname=None,
    x=TIME.copy(),
):
    y_preds = starccato_model.generate(z=z_posterior)
    ypred_qtls = pointwise_ci(y_preds, ci=0.9)
    yrecn_qtls, yrecn_cov = None, None

    posterior_label = "Posterior"
    if y_true is not None:
        ypred_cov = coverage_probability(ypred_qtls, y_true)
        y_recon = starccato_model.reconstruct(y_true, n_reps=nsamps)
        yrecn_qtls = pointwise_ci(y_recon, ci=0.9)
        yrecn_cov = coverage_probability(yrecn_qtls, y_true)
        posterior_label = f"Posterior (coverage: {ypred_cov:.0%})"

    plt.figure(figsize=(5, 3.5))
    plt.plot(
        x, y_obs, label="Observed", color="black", lw=2, zorder=-10, alpha=0.2
    )
    if y_true is not None:
        plt.plot(x, y_true, label="True", color="black", lw=2, zorder=-100)
    ax = plt.gca()

    add_quantiles(ax, ypred_qtls, posterior_label, "tab:orange")
    if yrecn_qtls is not None:
        add_quantiles(
            ax,
            yrecn_qtls,
            f"Reconstruction (coverage: {yrecn_cov:.0%})",
            "tab:green",
        )

    ax.set_xlim(x.min(), x.max())
    ax.set_xlabel("Time [s]")

    plt.tight_layout()
    plt.legend(frameon=False)
    if fname is not None:
        plt.savefig(fname)
