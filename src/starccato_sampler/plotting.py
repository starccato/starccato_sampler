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


def sampler_diagnostic_plots(inf_object: az.InferenceData, outdir: str):
    plot_trace(inf_object, outdir)
    plot_1d_marginals(inf_object, outdir)


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


def plot_1d_marginals(
    inf_object: az.InferenceData, outdir: str, color="lightblue"
):
    zdim = inf_object.posterior["z"].shape[-1]
    zpost = inf_object.posterior["z"].stack(sample=("chain", "draw")).values.T
    figsize = (3, zdim * 0.25)
    # plot a vertical violin plot of the zpost
    fig, ax = plt.subplots(figsize=figsize)
    boxplot = ax.boxplot(
        zpost,
        vert=False,
        patch_artist=True,
        showfliers=False,
        capwidths=0,
        widths=0.1,
        medianprops=dict(
            marker="s",
            markersize=4,
            markerfacecolor=color,
            markeredgecolor=color,
            lw=0,
        ),
        boxprops=dict(facecolor=color, edgecolor=color),
        whiskerprops=dict(color=color),
        showmeans=False,
    )
    ax.set_ylabel("Z idx")
    ax.set_xlabel("Z value")
    if "true_latent" in inf_object.sample_stats:
        true_z = inf_object.sample_stats["true_latent"].to_numpy().ravel()
        # draw a short line at the true z value for each box
        box_height = (
            boxplot["medians"][0].get_ydata()[1]
            - boxplot["medians"][0].get_ydata()[0]
        ) * 3
        for i, true_val in enumerate(
            true_z, start=1
        ):  # Boxplots are indexed from 1
            ax.vlines(
                x=true_val,
                ymin=i - box_height / 2,
                ymax=i + box_height / 2,
                color="tab:orange",
                linewidth=2,
                zorder=10,
            )
    plt.savefig(os.path.join(outdir, "1d_marginals.png"))
    plt.close()


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
