import harmonic as hm
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def compute_harmonic_mean_evidence(chains: np.ndarray, lnprob: np.ndarray):
    nchains, nsamples, ndim = chains.shape
    assert lnprob.shape == (
        nchains,
        nsamples,
    ), f"lnprob should have shape (nchains={nchains}, nsamples={nsamples}), got {lnprob.shape}"

    # Instantiate harmonic's chains class
    hm_chains = hm.Chains(ndim)
    hm_chains.add_chains_3d(chains, lnprob)

    # Split the chains into the ones which will be used to train the machine learning model and for inference
    chains_train, chains_infer = hm.utils.split_data(
        hm_chains, training_proportion=0.5
    )

    # Select a machine learning model and train it
    model = hm.model.RealNVPModel(ndim, standardize=True, temperature=0.8)
    model.fit(chains_train.samples, verbose=True, epochs=20)

    # samples = chains.reshape((-1, ndim))
    # samp_num = samples.shape[0]
    # flow_samples = model.sample(samp_num)
    # hm.utils.plot_getdist_compare(samples, flow_samples)
    # plt.show()

    # Instantiate harmonic's evidence class
    ev = hm.Evidence(chains_infer.nchains, model)

    # Pass the evidence class the inference chains and compute the evidence!
    ev.add_chains(chains_infer)
    return ev.compute_ln_evidence()
