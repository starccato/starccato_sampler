import os

import arviz as az
import corner
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from starccato_jax import StarccatoVAE
from starccato_jax.credible_intervals import coverage_probability, pointwise_ci
from starccato_jax.plotting import add_quantiles


def sampler_diagnostic_plots(inf_object: az.InferenceData, outdir: str):
    # Plot the trace plot
    axes = az.plot_trace(inf_object, var_names=["z"])
    if "true_latent" in inf_object.sample_stats:
        true_z = inf_object.sample_stats["true_latent"].to_numpy().ravel()

        if len(true_z) > 2:
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            for i, _z in enumerate(true_z):
                axes[0, 0].axvline(
                    _z, color=colors[i % len(colors)], ls="--", alpha=0.3
                )
    axes[0, 0].set_ylim(bottom=0)
    plt.savefig(os.path.join(outdir, "trace_plot.png"))
    plt.close()

    # # Plot the corner plot
    # corner.corner(inf_object, var_names=["z"])
    # plt.savefig(os.path.join(outdir, "corner_plot.png"))
    # plt.close()


def plot_ci(
    y_obs: jnp.ndarray,
    z_posterior: jnp.ndarray,
    starccato_vae: StarccatoVAE,
    y_true: jnp.ndarray = None,
    nsamps=100,
    fname=None,
):
    z_samples = z_posterior[
        np.random.choice(z_posterior.shape[0], nsamps, replace=False)
    ]
    y_preds = starccato_vae.generate(z=z_samples)
    ypred_qtls = pointwise_ci(y_preds, ci=0.9)
    yrecn_qtls, yrecn_cov = None, None

    if y_true is not None:
        ypred_cov = coverage_probability(ypred_qtls, y_true)
        y_recon = starccato_vae.reconstruct(y_true, n_reps=nsamps)
        yrecn_qtls = pointwise_ci(y_recon, ci=0.9)
        yrecn_cov = coverage_probability(yrecn_qtls, y_true)

    plt.figure(figsize=(5, 3.5))
    plt.plot(
        y_obs, label="Observed", color="black", lw=2, zorder=-10, alpha=0.2
    )
    if y_true is not None:
        plt.plot(y_true, label="True", color="black", lw=2, zorder=0)
    ax = plt.gca()
    add_quantiles(
        ax, ypred_qtls, f"Posterior (coverage: {ypred_cov:.0%})", "tab:orange"
    )
    if yrecn_qtls is not None:
        add_quantiles(
            ax,
            yrecn_qtls,
            f"Reconstruction (coverage: {yrecn_cov:.0%})",
            "tab:green",
        )

    ax.set_xlim(0, len(y_obs))

    plt.legend(frameon=False)
    if fname is not None:
        plt.savefig(fname)
