import os

import jax
import pytest
from starccato_jax import StarccatoVAE

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
    sample(
        true_signal,
        outdir=outdir,
        num_warmup=150,
        num_samples=1000,
        num_chains=1,
        verbose=True,
        stepping_stone_lnz=True,
        gss_lnz=True,
        truth=dict(signal=true_signal, latent=true_z),
    )
    assert os.path.exists(outdir)
    assert os.path.exists(os.path.join(outdir, "inference.nc"))
    assert os.path.exists(os.path.join(outdir, "ci_plot.png"))
    assert os.path.exists(os.path.join(outdir, "1d_marginals.png"))
