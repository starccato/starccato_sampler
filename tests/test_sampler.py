from starccato_jax.data import load_training_data

from starccato_sampler.sampler import sample


def test_sampler(outdir):
    _, val_data = load_training_data(train_fraction=0.8)
    sample(
        val_data[0],
        outdir=outdir,
        num_warmup=5,
        num_samples=100,
        num_chains=1,
        num_temps=1,
        verbose=True,
    )
