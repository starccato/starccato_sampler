import warnings

import bilby
import numpy as np
from bilby.core.prior import DeltaFunction, Normal, PriorDict, Uniform
from jax.random import PRNGKey
from starccato_jax import StarccatoVAE
from starccato_jax.data import load_training_data

# pip install scipy==1.14.0


vae = StarccatoVAE()
train_dat, val_dat = load_training_data()

rng = PRNGKey(0)

warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

# Set the duration and sampling frequency of the data segment that we're going
# to inject the signal into.
# These are fixed by the resolution in the injection file that we are using.

sampling_frequency = 4096
n_timestamps = 256
duration = n_timestamps / sampling_frequency
time = np.linspace(0, duration, n_timestamps)
t0 = time[53]  # This is the time of the core-bounce

LUMIN_DIST = 10  # kpc
GEOCENT_TIME = 1126259642.413

# Specify the output directory and the name of the simulation.
outdir = "outdir"
label = "supernova"
bilby.core.utils.setup_logger(outdir=outdir, label=label)

np.random.seed(170801)


def get_latent_vector_prior():
    priors = {}
    for i in range(vae.latent_dim):
        priors[f"z_{i}"] = Normal(0, 1, name=f"z_{i}")
    return priors


prior = PriorDict(
    dict(
        **get_latent_vector_prior(),
        luminosity_distance=Uniform(2, 20, "luminosity_distance", unit="kpc"),
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
    """
    A source model that reads a simulation from a text file.

    This was originally intended for use with supernova simulations, but can
    be applied to any source class.

    Parameters
    ----------
    frequency_array: array-like
        Unused (but required by the source model interface)

    Returns
    -------
    dict:
        A dictionary containing the plus and cross components of the signal.
    """
    # print all args
    rng = kwargs["rng"]
    # print(f"z: {z}, {z.shape}")
    waveform = vae.generate(z=z, rng=rng)[0]
    # print(f"waveform: {waveform}, {waveform.shape}")

    # waveforms generated at 10kpc, so scale to the luminosity distance
    scaling = 1e-21 * (10.0 / luminosity_distance)

    # idk if the signal is hcross/hplus, so we just duplicate for now... #TODO
    h_plus = scaling * waveform
    return {"plus": h_plus, "cross": h_plus}


def parameter_conversion(parameters):
    new_params = {}
    new_params["luminosity_distance"] = parameters["luminosity_distance"]
    z = np.array([parameters[f"z_{i}"] for i in range(vae.latent_dim)])
    # ennsure z is of shape (1, latent_dim)
    if z.shape != (1, vae.latent_dim):
        z = z.reshape(1, vae.latent_dim)

    new_params["z"] = z

    return new_params, parameters


# Create the waveform_generator using a supernova source function
waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    time_domain_source_model=supernova,
    parameters=injection_parameters,
    parameter_conversion=parameter_conversion,
    waveform_arguments=dict(rng=rng),
)

# Set up interferometers (LIGO-Hanford (H1) and LIGO-Livingston (L1))
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

import matplotlib.pyplot as plt
from bilby.gw import utils as gwutils

DATA_COL = "tab:gray"
SIGNAL_COL = "tab:orange"
PSD_COL = "black"


def plot_freq_domain(ifo: bilby.gw.detector.Interferometer, freq_signal):
    fig, ax = plt.subplots()
    df = (
        ifo.strain_data.frequency_array[1] - ifo.strain_data.frequency_array[0]
    )
    asd = gwutils.asd_from_freq_series(
        freq_data=ifo.strain_data.frequency_domain_strain, df=df
    )

    ax.loglog(
        ifo.strain_data.frequency_array[ifo.strain_data.frequency_mask],
        asd[ifo.strain_data.frequency_mask],
        color=DATA_COL,
        label=f"{ifo.name} Data",
        alpha=0.5,
        lw=3,
    )
    ax.loglog(
        ifo.strain_data.frequency_array[ifo.strain_data.frequency_mask],
        ifo.amplitude_spectral_density_array[ifo.strain_data.frequency_mask],
        color=PSD_COL,
        lw=1.0,
        label=ifo.name + " ASD",
    )

    signal_asd = gwutils.asd_from_freq_series(freq_data=freq_signal, df=df)

    ax.loglog(
        ifo.strain_data.frequency_array[ifo.strain_data.frequency_mask],
        signal_asd[ifo.strain_data.frequency_mask],
        color=SIGNAL_COL,
        label=f'Signal (SNR: {ifo.meta_data["optimal_SNR"]:.2f})',
    )
    ax.grid(True)
    ax.set_ylabel(r"Strain [strain/$\sqrt{\rm Hz}$]")
    ax.set_xlabel(r"Frequency [Hz]")
    ax.legend(loc="best")


def plot_time_domain(ifo, time_signal):
    fig, ax = plt.subplots()
    strain = ifo.strain_data.time_domain_strain
    t0 = ifo.strain_data.start_time
    x = ifo.strain_data.time_array - t0
    xlabel = f"GPS time [s] - {t0}"
    # unroll raw data
    strain = np.roll(strain, 55)
    ax.plot(
        x, strain, color=DATA_COL, label=f"{ifo.name} Data", alpha=0.5, lw=3
    )
    ax.plot(
        x,
        time_signal,
        color=SIGNAL_COL,
        label=f'Signal (SNR: {ifo.meta_data["optimal_SNR"]:.2f})',
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Strain")
    ax.legend()
    fig.tight_layout()
    return ax


plot_freq_domain(ifos[0], injection_strain["plus"])
plot_time_domain(ifos[0], injection_strain_time["plus"])

signals = []
for i in range(100):
    sample = prior.sample()
    s = waveform_generator.time_domain_strain(sample)["plus"]
    signals.append(s)

signal_ci = np.quantile(np.array(signals), [0.05, 0.5, 0.95], axis=0)

t0 = ifos[0].strain_data.start_time
x = ifos[0].strain_data.time_array - t0
ax = plot_time_domain(ifos[0], injection_strain_time["plus"])
ax.fill_between(
    x, signal_ci[0], signal_ci[2], alpha=0.25, color="tab:blue", label="Prior"
)
ax.legend()

# Run sampler.
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator
)

result = bilby.run_sampler(
    likelihood=likelihood,
    priors=prior,
    sampler="emcee",
    outdir=outdir,
    label=label,
    clean=True,
)
