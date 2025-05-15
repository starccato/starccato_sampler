import warnings
from dataclasses import dataclass

import bilby
import matplotlib.pyplot as plt
import numpy as np
from bilby.core.prior import DeltaFunction, Normal, PriorDict, Uniform
from bilby.gw import utils as gwutils
from jax import numpy as jnp
from jax.random import PRNGKey
from starccato_jax import StarccatoVAE
from starccato_jax.data import load_training_data


@dataclass
class AnalysisData:
    frequencies: jnp.ndarray
    timepoints: jnp.ndarray
    psd: jnp.ndarray
    timeseries_data: jnp.ndarray
    true_timeseries: jnp.ndarray
    freqseries_data: jnp.ndarray
    true_freqseries: jnp.ndarray
    fmask: jnp.ndarray

    @property
    def df(self):
        return self.frequencies[1] - self.frequencies[0]


def generate_analysis_data():
    vae = StarccatoVAE()
    train_dat, val_dat = load_training_data()

    rng = PRNGKey(0)

    warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

    sampling_frequency = 4096
    n_timestamps = 512
    duration = n_timestamps / sampling_frequency
    time = np.linspace(0, duration, n_timestamps)
    t0 = time[53]

    LUMIN_DIST = 30  # kpc
    GEOCENT_TIME = 1126259642.413

    outdir = "outdir"
    label = "supernova"
    bilby.core.utils.setup_logger(outdir=outdir, label=label)

    np.random.seed(42)

    def get_latent_vector_prior():
        priors = {}
        for i in range(vae.latent_dim):
            priors[f"z_{i}"] = Normal(0, 1, name=f"z_{i}")
        return priors

    prior = PriorDict(
        dict(
            **get_latent_vector_prior(),
            luminosity_distance=Uniform(
                2, 40, "luminosity_distance", unit="kpc"
            ),
            ra=Uniform(0, 2 * np.pi, "ra", boundary="periodic"),
            dec=Uniform(-np.pi / 2, np.pi / 2, "dec", boundary="periodic"),
            geocent_time=Uniform(
                GEOCENT_TIME - 1, GEOCENT_TIME + 1, "geocent_time", unit="s"
            ),
            psi=DeltaFunction(name="psi", unit="radian", peak=0.0),
        ),
    )

    injection_parameters = dict(
        luminosity_distance=LUMIN_DIST,  # kpc
        geocent_time=GEOCENT_TIME,
        ra=0,
        dec=0,
        psi=0,
        **PriorDict(get_latent_vector_prior()).sample(1),
    )

    def supernova(freq_array, luminosity_distance, z, **kwargs):
        rng = kwargs["rng"]
        waveform = vae.generate(z=z, rng=rng)[0]
        scaling = 1e-21 * (10.0 / luminosity_distance)
        h_plus = scaling * waveform
        return {"plus": h_plus, "cross": h_plus}

    def parameter_conversion(parameters):
        new_params = {}
        new_params["luminosity_distance"] = parameters["luminosity_distance"]
        z = np.array([parameters[f"z_{i}"] for i in range(vae.latent_dim)])
        if z.shape != (1, vae.latent_dim):
            z = z.reshape(1, vae.latent_dim)
        new_params["z"] = z
        return new_params, parameters

    class WaveformGenerator(bilby.gw.waveform_generator.WaveformGenerator):
        def _calculate_strain(
            self,
            model,
            model_data_points,
            transformation_function,
            transformed_model,
            transformed_model_data_points,
            parameters,
        ):
            if parameters is not None:
                self.parameters = parameters
            if model is not None:
                model_strain = self._strain_from_model(
                    model_data_points, model
                )
            elif transformed_model is not None:
                model_strain = self._strain_from_transformed_model(
                    transformed_model_data_points,
                    transformed_model,
                    transformation_function,
                )
            else:
                raise RuntimeError("No source model given")
            self._cache["waveform"] = model_strain
            self._cache["parameters"] = self.parameters.copy()
            self._cache["model"] = model
            self._cache["transformed_model"] = transformed_model
            return model_strain

    waveform_generator = WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        time_domain_source_model=supernova,
        parameters=injection_parameters,
        parameter_conversion=parameter_conversion,
        waveform_arguments=dict(rng=rng),
    )

    ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=injection_parameters["geocent_time"],
    )

    injection_strain_time = waveform_generator.time_domain_strain(
        injection_parameters
    )
    injection_strain = waveform_generator.frequency_domain_strain(
        injection_parameters
    )

    ifos.inject_signal(
        injection_polarizations=injection_strain,
        parameters=injection_parameters,
        raise_error=False,
    )

    signals_td = []
    signals_fd = []
    df = (
        ifos[0].strain_data.frequency_array[1]
        - ifos[0].strain_data.frequency_array[0]
    )
    fmask = ifos[0].strain_data.frequency_mask
    for i in range(100):
        sample = prior.sample()
        td = waveform_generator.time_domain_strain(sample)["plus"]
        fd = waveform_generator.frequency_domain_strain(sample)["plus"]
        asd = gwutils.asd_from_freq_series(freq_data=fd, df=df)[fmask]
        signals_td.append(td)
        signals_fd.append(asd)

    td_ci = np.quantile(np.array(signals_td), [0.05, 0.5, 0.95], axis=0)
    fd_ci = np.quantile(np.array(signals_fd), [0.05, 0.5, 0.95], axis=0)
    t0 = ifos[0].strain_data.start_time
    t = ifos[0].strain_data.time_array - t0
    f = ifos[0].strain_data.frequency_array[ifos[0].strain_data.frequency_mask]

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))

    plot_time_domain(ifos[0], injection_strain_time["plus"], ax=ax[0])
    ax[0].fill_between(
        t, td_ci[0], td_ci[2], alpha=0.25, color="tab:red", label="Prior"
    )

    plot_freq_domain(ifos[0], injection_strain["plus"], ax=ax[1])
    ax[1].fill_between(
        f,
        fd_ci[0],
        fd_ci[2],
        alpha=0.25,
        color="tab:red",
        label="Prior",
    )
    ax[0].set_xlim(min(t), max(t))
    ax[1].set_xlim(200, 2048)
    ax[1].legend(loc="lower left", frameon=False)
    ax[1].grid(False)
    plt.tight_layout()
    plt.savefig("injection.png")
    return AnalysisData(
        frequencies=jnp.array(
            ifos[0].strain_data.frequency_array[
                ifos[0].strain_data.frequency_mask
            ]
        ),
        timepoints=jnp.array(ifos[0].strain_data.time_array),
        psd=jnp.array(
            ifos[0].amplitude_spectral_density_array[
                ifos[0].strain_data.frequency_mask
            ]
        ),
        timeseries_data=jnp.array(ifos[0].strain_data.time_domain_strain),
        true_timeseries=jnp.array(injection_strain_time["plus"]),
        freqseries_data=jnp.array(
            ifos[0].strain_data.frequency_domain_strain[
                ifos[0].strain_data.frequency_mask
            ]
        ),
        true_freqseries=jnp.array(injection_strain["plus"]),
        fmask=jnp.array(ifos[0].strain_data.frequency_mask),
    )


def plot_freq_domain(
    ifo: bilby.gw.detector.Interferometer, freq_signal, ax=None
):
    if ax is None:
        fig, ax = plt.subplots()
    fig = ax.get_figure()
    df = (
        ifo.strain_data.frequency_array[1] - ifo.strain_data.frequency_array[0]
    )
    asd = gwutils.asd_from_freq_series(
        freq_data=ifo.strain_data.frequency_domain_strain, df=df
    )

    ax.loglog(
        ifo.strain_data.frequency_array[ifo.strain_data.frequency_mask],
        asd[ifo.strain_data.frequency_mask],
        color="tab:gray",
        label=f"{ifo.name} Data",
        alpha=0.5,
        lw=3,
    )
    ax.loglog(
        ifo.strain_data.frequency_array[ifo.strain_data.frequency_mask],
        ifo.amplitude_spectral_density_array[ifo.strain_data.frequency_mask],
        color="black",
        lw=1.0,
        label=ifo.name + " ASD",
    )

    signal_asd = gwutils.asd_from_freq_series(freq_data=freq_signal, df=df)

    ax.loglog(
        ifo.strain_data.frequency_array[ifo.strain_data.frequency_mask],
        signal_asd[ifo.strain_data.frequency_mask],
        color="tab:orange",
        label=f'Signal (SNR: {ifo.meta_data["optimal_SNR"]:.2f})',
    )
    ax.grid(True)
    ax.set_ylabel(r"Strain [strain/$\sqrt{\rm Hz}$]")
    ax.set_xlabel(r"Frequency [Hz]")


def plot_time_domain(ifo, time_signal, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    fig = ax.get_figure()
    strain = ifo.strain_data.time_domain_strain
    t0 = ifo.strain_data.start_time
    x = ifo.strain_data.time_array - t0
    xlabel = f"GPS time [s] - {t0}"
    strain = np.roll(strain, 55)
    ax.plot(
        x, strain, color="tab:gray", label=f"{ifo.name} Data", alpha=0.5, lw=3
    )
    ax.plot(
        x,
        time_signal,
        color="tab:orange",
        label=f'Signal (SNR: {ifo.meta_data["optimal_SNR"]:.2f})',
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Strain")
    fig.tight_layout()
    return ax


if __name__ == "__main__":
    data = generate_analysis_data()
    print(data)
