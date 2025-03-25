import os
from typing import List

import arviz as az
import matplotlib.pyplot as plt


def plot_1d_marginals(
    inf_object: az.InferenceData, outdir: str, color="lightblue"
):
    zpost = inf_object.posterior["z"].stack(sample=("chain", "draw")).values.T
    if "true_latent" in inf_object.sample_stats:
        true_z = inf_object.sample_stats["true_latent"].to_numpy().ravel()
    else:
        true_z = None

    fig, ax = _plot_marginals(zpost, trues=true_z, color=color)
    plt.savefig(os.path.join(outdir, "1d_marginals.png"))
    plt.close()


def _plot_marginals(samples, trues=None, color="tab:orange"):
    dim = samples.shape[-1]
    figsize = (3, dim * 0.25)
    # plot a vertical violin plot of the zpost
    fig, ax = plt.subplots(figsize=figsize)
    boxplot = ax.boxplot(
        samples,
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
    if trues is not None:
        # draw a short line at the true z value for each box
        box_height = (
            boxplot["medians"][0].get_ydata()[1]
            - boxplot["medians"][0].get_ydata()[0]
        ) * 3
        for i, true_val in enumerate(
            trues, start=1
        ):  # Boxplots are indexed from 1
            ax.vlines(
                x=true_val,
                ymin=i - box_height / 2,
                ymax=i + box_height / 2,
                color="tab:orange",
                linewidth=2,
                zorder=10,
            )

    return fig, ax
