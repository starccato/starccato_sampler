# system functions that are always useful to have
import os
import sys
import time

import dynesty
import jax
import jax.numpy as jnp

# plotting
import matplotlib

# basic numeric setup
import numpy as np
from dynesty import plotting as dyplot
from matplotlib import pyplot as plt
from scipy.stats import norm
from starccato_jax import StarccatoVAE

from starccato_sampler.sampler import sample

plt.style.use("seaborn-v0_8-poster")

rstate = np.random.default_rng(56101)

MODEL = StarccatoVAE()
RNG = jax.random.PRNGKey(0)

NOISE_SIGMA = 0.1
OUTDIR = "output_dynesty"
os.makedirs(OUTDIR, exist_ok=True)


# log-likelihood
@jax.jit
def loglike(theta):
    model = MODEL.generate(theta)
    inv_sigma2 = 1.0 / (NOISE_SIGMA**2)
    return -0.5 * (np.sum((y - model) ** 2 * inv_sigma2 - np.log(inv_sigma2)))


def prior_transform(utheta):
    # Transform from uniform to standard normal... i think?
    return norm.ppf(uthetas)


def plot_marginals(samples, trues=None, color="tab:orange"):
    dim = samples.shape[-1]
    figsize = (3, dim * 0.25)
    fig, ax = plt.subplots(figsize=figsize)

    # Create the horizontal boxplot
    bp = ax.boxplot(
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
        positions=range(1, dim + 1),
        boxprops=dict(facecolor=color, edgecolor=color),
        whiskerprops=dict(color=color),
        showmeans=False,
    )

    ax.set_ylabel("Z idx")
    ax.set_xlabel("Z value")

    # Add true value markers using the default y-positions (1-indexed)
    if trues is not None:
        for i, true_val in enumerate(trues.ravel().T, start=1):
            ax.vlines(
                x=true_val,
                ymin=i - 0.4,  # adjust the vertical span if needed
                ymax=i + 0.4,
                color="tab:orange",
                linewidth=2,
                zorder=10,
            )
    return fig, ax


def plot_data(y, y_true, z_samples=None, fname="data.png"):
    x = np.arange(len(y))
    plt.figure(figsize=(7, 4))
    plt.plot(x, y_true, color="k", lw=2, label="True Model")
    plt.scatter(x, y, c="gray", label="Data", marker="|")
    plt.xlabel(r"$X$")
    plt.ylabel(r"$Y$")

    if z_samples is not None:
        model_preds = MODEL.generate(z=z_samples, rng=RNG)
        ci = np.percentile(model_preds, [16, 84], axis=0)
        plt.fill_between(
            x,
            ci[0],
            ci[1],
            color="tab:orange",
            alpha=0.5,
            label="Model Prediction",
            zorder=10,
        )

    plt.xlim(x.min(), x.max())
    plt.tight_layout()
    plt.savefig(fname)


# define the true model
z_true = jnp.array(np.random.normal(0, 1, size=(1, MODEL.latent_dim)))
y_true = MODEL.generate(z=z_true, rng=RNG)[0]
N = len(y_true)

# generate mock data
yerr = rstate.normal(size=N) * NOISE_SIGMA
y = y_true + yerr
x = np.arange(N)

plot_data(y, y_true, fname=f"{OUTDIR}/data.png")

# dsampler = dynesty.NestedSampler(
#     loglike,
#     prior_transform,
#     ndim=MODEL.latent_dim,
#     rstate=rstate,
#     nlive=5,
#     gradient=jax.grad(loglike),
#     bound="none",
#     sample="hslice",
#     slices=5,
# )
# dsampler.run_nested(dlogz=0.01)
# dres = dsampler.results
#
# fig, axes = dyplot.runplot(dres, color="red", logplot=True)
# fig.tight_layout()
# fig.savefig(f"{OUTDIR}/diagnostics.png")
#
# fig, ax = plot_marginals(dres.samples, trues=z_true, color="tab:red")
# fig.tight_layout()
# fig.savefig(f"{OUTDIR}/marginals.png")
#
# plot_data(
#     y,
#     y_true,
#     z_samples=dres.samples,
#     fname=f"{OUTDIR}/posterior_predictive.png",
# )
#
# print(dres.summary())


sample(
    data=y,
    model=MODEL,
    rng_int=0,
    outdir=OUTDIR,
    noise_sigma=NOISE_SIGMA,
    stepping_stone_lnz=True,
    n_temps=16,
)
