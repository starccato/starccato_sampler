import os

import jax
import pytest
from starccato_jax import StarccatoPCA, StarccatoVAE

from starccato_sampler.plotting import plot_pe_comparison
from starccato_sampler.sampler import sample

RNG = jax.random.PRNGKey(0)


@pytest.fixture
def injection():
    vae = StarccatoVAE()
    # true_z = jax.random.normal(RNG, (1, vae.latent_dim))
    true_z = jax.numpy.zeros((1, vae.latent_dim))
    true_signal = vae.generate(z=true_z)[0]
    return true_signal, true_z.ravel()


def test_sampler(injection, outdir):
    true_signal, true_z = injection
    models = [StarccatoVAE(), StarccatoPCA()]
    for model in models:
        sample(
            true_signal,
            model,
            rng_int=0,
            outdir=f"{outdir}/{model}",
            num_warmup=500,
            num_samples=5000,
            num_chains=1,
            verbose=True,
            ns_lnz=True,
            stepping_stone_lnz=False,
            truth=dict(signal=true_signal, latent=true_z),
        )
        assert os.path.exists(outdir)
        assert os.path.exists(f"{outdir}/{model}/inference.nc")
        assert os.path.exists(f"{outdir}/{model}/ci_plot.png")
        assert os.path.exists(f"{outdir}/{model}/1d_marginals.png")

    plot_pe_comparison(
        data_1d=true_signal,
        true_1d=true_signal,
        result_fnames=[f"{outdir}/{model}/inference.nc" for model in models],
        labels=[model.__class__.__name__ for model in models],
        colors=["tab:blue", "tab:red"],
        fname=f"{outdir}/comparison.pdf",
    )
