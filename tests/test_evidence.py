import dataclasses
import os
from typing import List

import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
from lartillot_gaussian import LartillotGaussianModel
from sample_lartillot import sample_lartillot
from jax import numpy as jnp
from starccato_sampler.evidence.gss_evidence import generate_reference_prior

from starccato_sampler.evidence import (  # compute_harmonic_mean_evidence,
    compute_gaussian_approx_evidence,
    compute_stepping_stone_evidence,
)
from starccato_sampler.utils import beta_spaced_samples
import jax

jax.config.update("jax_enable_x64", True)

RNG = random.PRNGKey(0)



@dataclasses.dataclass
class LnZEstimate:
    ln_z: float
    err: float
    label: str

    def __repr__(self):
        return f"{self.label}: {self.ln_z:.3f} +/- {self.err:.3f}"






def test_stepping_stone_evidence(outdir):
    d = 90
    lartillot_model = LartillotGaussianModel(d=d, v=0.01)
    rng = random.PRNGKey(0)
    ntemps = 64
    betas = beta_spaced_samples(ntemps, 0.3, 1)
    tempd_chains = lartillot_model.generate_chains(1000, betas)

    ss_lnz = LnZEstimate(
        *compute_stepping_stone_evidence(
            tempd_chains.lnl.T, betas, outdir, rng
        ),
        "Stepping Stone",
    )

    post_samples = lartillot_model.simulate_posterior_samples(5000, beta=1.0)
    gauss_lnz = LnZEstimate(
        *compute_gaussian_approx_evidence(
            ref_sample=np.zeros(d),
            lnl_ref=lartillot_model.log_likelihood(np.zeros((1, d))),
            posterior_samples=post_samples,
            n_bootstraps=30,
        ),
        "Gaussian Approx",
    )



    # chain = lartillot_model.generate_chains(1000, nchains=100)
    # harmonic_lnz = LnZEstimate(
    #     *compute_harmonic_mean_evidence(chains.samples, chains.lnl), "Harmonic Mean"
    # )

    lnzs = [ss_lnz, gauss_lnz]
    print(lnzs)
    plot_estimate(lartillot_model.lnZ, lnzs, outdir)

    plot_path = os.path.join(outdir, "lnz_comparison.png")

    for lnz in lnzs:
        np.testing.assert_allclose(
            lnz.ln_z,
            lartillot_model.lnZ,
            atol=0.3,
            err_msg=f"{lnz} not close to {lartillot_model.lnZ} within tolerance",
        )

    assert os.path.exists(plot_path), "Plot file should be created"

from numpyro.infer import MCMC, NUTS
from numpyro.distributions import Normal


def test_manual_sampling(outdir):
    d, v = 4, 0.01

    true_lnz = np.log((v/ (1.0+v)) ** (d/2.0))

    lartillot_model = LartillotGaussianModel(d=d, v=v)
    ntemps = 16
    betas = beta_spaced_samples(ntemps, 0.3, 1)

    chains = [sample_lartillot(d, v, beta=beta) for beta in betas]
    lnls = jnp.array([chain.get_samples(group_by_chain=True)["untempered_loglike"][0] for chain in chains]).T
    ss_lnz = LnZEstimate(
        *compute_stepping_stone_evidence(lnls, betas, f"{outdir}/lartillot_ss.png", rng=RNG), "Stepping Stone"
    )


    z_samples = chains[-1].get_samples()["z"]
    print(z_samples.shape)

    muz, sigz = z_samples.mean(0), z_samples.std(0)
    ref_pri = generate_reference_prior(z_samples)
    print(f"ref pri dim: {len(ref_pri)}")
    print(muz, sigz)
    print(f"ref pri: {ref_pri}")

    gss_chains = [sample_lartillot(d, v, beta=beta, reference_prior=ref_pri) for beta in betas]
    lnls = jnp.array([chain.get_samples(group_by_chain=True)["untempered_loglike"][0] for chain in gss_chains]).T
    gss_lnz = LnZEstimate(
        *compute_stepping_stone_evidence(lnls, betas, f"{outdir}/lartillot_gss.png", rng=RNG), "GSS"
    )

    lnzs = [ss_lnz, gss_lnz]
    print(lnzs)
    print(true_lnz)
    plot_estimate(true_lnz, lnzs, outdir)



def plot_estimate(
    true_lnZ: float,
    lnzs: List[LnZEstimate],
    outdir: str = "out",
):
    plt.figure(figsize=(2 * len(lnzs), 3))

    for i, lnz in enumerate(lnzs):
        plt.errorbar(
            i,
            lnz.ln_z,
            yerr=lnz.err,
            fmt="o",
            color="tab:orange",
            label=lnz.label,
        )

    ylims = plt.ylim()

    # max_err = max([lnz.err for lnz in lnzs])
    #
    # # ensure ylims at least +- 2 from true_lnZ
    # if true_lnZ - max_err < ylims[0]:
    #     plt.ylim(true_lnZ - max_err, true_lnZ + max_err)
    #
    # plt.ylim(true_lnZ - max_err, true_lnZ + max_err)
    # set tick labels
    plt.xticks(range(len(lnzs)), [lnz.label for lnz in lnzs])
    plt.axhline(true_lnZ, color="r", label="True LnZ")
    plt.ylabel("LnZ")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "lnz_comparison.png"))
