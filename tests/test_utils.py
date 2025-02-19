import matplotlib.pyplot as plt

from starccato_sampler.utils import beta_spaced_samples


def test_version():
    from starccato_sampler import __version__

    assert isinstance(__version__, str)


def test_beta_samples(outdir):
    plt.plot(beta_spaced_samples(100, 0.3, 1))
    plt.savefig(f"{outdir}/test_beta_samples.png")
