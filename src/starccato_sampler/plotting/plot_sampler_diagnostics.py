import os
from typing import List

import arviz as az
import corner
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from starccato_jax import StarccatoVAE
from starccato_jax.credible_intervals import coverage_probability, pointwise_ci
from starccato_jax.plotting import TIME, add_quantiles
from starccato_jax.starccato_model import StarccatoModel

from .plot_1d_marginals import plot_1d_marginals


def plot_sampler_diagnostics(
    inf_object: az.InferenceData, outdir: str, plot_corner: bool = False
):
    plot_trace(inf_object, outdir)
    plot_1d_marginals(inf_object, outdir)
    if plot_corner:
        plot_corner(inf_object, outdir)


def plot_trace(inf_object: az.InferenceData, outdir: str):
    # Plot the trace plot
    axes = az.plot_trace(inf_object, var_names=["z"])
    zdim = inf_object.posterior["z"].shape[-1]

    if "true_latent" in inf_object.sample_stats:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        true_z = inf_object.sample_stats["true_latent"].to_numpy().ravel()
        for i, _z in enumerate(true_z):
            axes[0, 0].axvline(
                _z, color=colors[i % len(colors)], ls="--", alpha=0.3
            )

    axes[0, 0].set_ylim(bottom=0)
    plt.savefig(os.path.join(outdir, "trace_plot.png"))
    plt.close()


def plot_corner(inf_object: az.InferenceData, outdir: str):
    corner.corner(inf_object, var_names=["z"])
    plt.savefig(os.path.join(outdir, "corner_plot.png"))
    plt.close()
