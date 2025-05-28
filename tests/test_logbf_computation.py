import arviz as az
import jax
import matplotlib.pyplot as plt
from starccato_jax.waveforms import StarccatoBlip, StarccatoCCSNe

from starccato_sampler.bayes_factor import compute_log_bf


def test_logbf(outdir):
    # Create the models
    blip_model = StarccatoBlip()
    ccsne_model = StarccatoCCSNe()

    # generate the data
    z = jax.numpy.zeros((1, blip_model.latent_dim))
    true_glitch = blip_model.generate(z=z)[0]
    true_signal = ccsne_model.generate(z=z)[0]

    bf_1 = compute_log_bf(
        data=true_signal,
        outdir=f"{outdir}/logbf/signal",
    )
    bf_2 = compute_log_bf(
        data=true_glitch,
        outdir=f"{outdir}/logbf/glitch",
    )

    print(f"Log Bayes Factor for signal: {float(bf_1):.2f}")
    print(f"Log Bayes Factor for glitch: {float(bf_2):.2f}")

    #
    #
    # # Plot the data for the signal and glitch
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ax[0].plot(true_signal)
    # ax[0].set_title(f"Signal (lnBF: {float(bf_1):.2f})")
    # ax[0].set_xlabel("Time [s]")
    # ax[0].set_ylabel("Amplitude")
    # ax[1].plot(true_glitch)
    # ax[1].set_title(f"Glitch (lnBF: {float(bf_2):.2f})")
    # ax[1].set_xlabel("Time [s]")
    # ax[1].set_ylabel("Amplitude")
    #
    # plt.tight_layout()
    # plt.savefig(f"{outdir}/logbf/BF.png")

    #
    # # load quantiles for signal and glitch
    # inf_object.sample_stats["quantiles"]
