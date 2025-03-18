from typing import List

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from starccato_jax.plotting import TIME


def plot_pe_comparison(
    data_1d: jax.numpy.ndarray,
    true_1d: jax.numpy.ndarray,
    result_fnames: List[str],
    labels: List[str],
    colors: List[str],
    fname: str = "comparison.pdf",
):
    results = [az.from_netcdf(f) for f in result_fnames]
    quantiles = [r.sample_stats["quantiles"].values for r in results]
    err = [q - true_1d for q in quantiles]

    fig = plt.figure(figsize=(4, 3))
    gs = GridSpec(
        2,
        1,
        height_ratios=[4, 1],
    )
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], sharex=ax0)
    ax0.set_xlim(TIME[0], TIME[-1])
    ax1.set_xlim(TIME[0], TIME[-1])
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Error")
    plt.subplots_adjust(hspace=0)

    ax0.plot(TIME, data_1d, label="Data", color="gray", lw=2, alpha=0.25)
    ax0.plot(TIME, true_1d, label="True", color="black", lw=1)
    for e, q, lbl, c in zip(err, quantiles, labels, colors):
        mse = jnp.mean(e[1] ** 2)
        _add_qtl(ax0, q, c, f"{lbl} (mse: {mse:.2e})")
        _add_qtl(ax1, e, c, lbl)
    ax0.legend(frameon=False)
    plt.savefig(fname, bbox_inches="tight")
    plt.close(fig)


def _add_qtl(
    ax: plt.Axes,
    qtl: jnp.ndarray,
    color: List[str],
    lbl: List[str],
):
    ax.fill_between(TIME, qtl[0], qtl[-1], color=color, alpha=0.25, lw=0)
    ax.plot(TIME, qtl[1], color=color, lw=1, ls="-", alpha=0.75, label=lbl)
