import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from jaxns import (
    Model,
    NestedSampler,
    Prior,
    bruteforce_evidence,
    load_results,
    plot_cornerplot,
    plot_diagnostics,
    save_results,
    summary,
)
from starccato_jax import StarccatoVAE

tfpd = tfp.distributions

starccato_model = StarccatoVAE()


# Define the SNR functions as in the original code
def match_filter_snr(h_dec, data, psd, df):
    return 4.0 * jnp.sum((jnp.conj(h_dec) * data) / psd * df).real


def optimal_snr(h_dec, psd, df):
    return 4.0 * jnp.sum((jnp.conj(h_dec) * h_dec) / psd * df).real


def prior():
    z_parts = []
    # Define each latent variable as a Normal prior
    for i in range(starccato_model.latent_dim):
        z_i = yield Prior(tfpd.Normal(loc=0.0, scale=5.0), name=f"z_{i}")
        z_parts.append(z_i)
    z = jnp.stack(z_parts, axis=0)
    return z


def likelihood(z, y_obs, psd, df, fmask):
    # Generate time-domain waveform from latent variables
    y_model_t = (
        starccato_model.generate(z=z) * 1e-21
    )  # Scaling from original code

    # Convert to frequency domain and apply frequency mask
    y_model = jnp.fft.rfft(y_model_t)
    y_model = y_model[fmask]

    # Calculate likelihood using observed data and PSD
    mf_snr = match_filter_snr(y_model, y_obs, psd, df)
    opt_snr = optimal_snr(y_model, psd, df)
    return mf_snr - 0.5 * opt_snr


def main():
    from data_generator import generate_analysis_data

    # Load data and model
    analysis_data = generate_analysis_data()
    starccato_model = StarccatoVAE()
    rng = jax.random.PRNGKey(0)

    # Create JAXNS model

    model = Model(
        prior_model=prior,
        log_likelihood=lambda z: likelihood(
            z=z,
            y_obs=analysis_data.freqseries_data,
            psd=analysis_data.psd,
            df=analysis_data.df,
            fmask=analysis_data.fmask,
        ),
    )

    # Sanity check model structure
    model.sanity_check(key=rng, S=100)

    # Configure and run nested sampling
    ns = NestedSampler(
        model=model,
        max_samples=1e4,
        k=10,  # Number of live points
        c=10,  # Number of MCMC steps per sample
        s=10,  # Number of slices per MCMC step
        # gradient_guided=True,
        verbose=True,
    )
    # ns = jax.jit(ns)

    # Run the sampler
    termination_reason, state = ns(jax.random.PRNGKey(42))
    # Get the results
    results = ns.to_results(termination_reason=termination_reason, state=state)

    # Optionally save the results to file
    save_results(results, "results.json")
    # To load the results back use this
    results = load_results("results.json")

    # Analyze results
    summary(results)
    plot_diagnostics(results, save_name="diagnostics.png")
    plot_cornerplot(results, save_name="cornerplot.png")
    print("Saved diagnostics and cornerplot to current directory")


if __name__ == "__main__":
    main()
