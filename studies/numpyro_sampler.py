import arviz as az
import jax
import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from data_generator import AnalysisData, generate_analysis_data
from jax.random import PRNGKey
from jaxtyping import Array, Float
from numpyro.infer import MCMC, NUTS
from starccato_jax import StarccatoVAE
from starccato_jax.starccato_model import StarccatoModel

jax.config.update("jax_enable_x64", True)


# import numpy as jnp


# @jax.jit
def match_filter_snr(
    h_dec: Array, data: Array, psd: Array, df: Float
) -> Float:
    return 4 * jnp.sum((jnp.conj(h_dec) * data) / psd * df).real


# @jax.jit
def optimal_snr(h_dec: Array, psd: Array, df: Float) -> Float:
    return 4 * jnp.sum(jnp.conj(h_dec) * h_dec / psd * df).real


# @jax.jit
def log_likelihood(
    h_dec: dict[str, Float[Array, " n_dim"]],
    data: list[Float[Array, " n_dim"]],
    psd: list[Float[Array, " n_dim"]],
    df: Float,
    # align_time: Float,
    **kwargs,
) -> Float:
    log_likelihood = 0.0

    # todo iterate over detectors
    # for data, psd in zip(datas, psds):

    # TODO: add response function
    # h_dec = detector.fd_response(freqs, h_sky, params) * align_time

    _match_filter_SNR = match_filter_snr(h_dec, data, psd, df)
    _optimal_SNR = optimal_snr(h_dec, psd, df)
    log_likelihood += _match_filter_SNR - _optimal_SNR / 2

    return log_likelihood


def _bayesian_model(
    y_obs: Array,
    psd: Array,
    starccato_model: StarccatoModel,
    rng: PRNGKey,
    fmask: Array,
    df: Float,
):
    """
    Bayesian model with tempering.

    Parameters:
      y_obs   : observed data.
      vae_data: model data (e.g. containing latent_dim).
      beta    : tempering parameter; beta=1 corresponds to full posterior.
    """

    # Define priors for the latent variables
    theta = numpyro.sample(
        "z", dist.Normal(0, 5).expand([starccato_model.latent_dim])
    )
    # Generate the signal
    y_model_t = starccato_model.generate(z=theta, rng=rng)
    y_model_t = 1e-21 * y_model_t
    # FFT model
    y_model = jnp.fft.rfft(y_model_t)

    # get rid of freq outside of fmin and fmax
    y_model = y_model[fmask]

    # Compute the untempered log–likelihood (Assuming Gaussian noise)
    lnl = log_likelihood(
        y_model,
        y_obs,
        psd,
        df,
    )

    # Temper the likelihood (the power likelihood)
    numpyro.factor("likelihood", lnl)

    # Save the untempered log–likelihood (for LnZ computation).
    # numpyro.deterministic("untempered_loglike", lnl)


def run_mcmc(
    analysis_data: AnalysisData,
    starccato_model: StarccatoModel,
    num_samples: int,
    num_warmup: int,
    num_chains: int,
    rng: PRNGKey,
    progress_bar: bool = False,
) -> MCMC:
    """
    Run MCMC sampling.

    Parameters:
      y_obs       : observed data.
      vae_data    : model data (e.g. containing latent_dim).
      num_samples : number of samples to draw.
      num_warmup  : number of warmup steps.
      num_chains  : number of chains.
      rng         : random number generator.
      beta        : tempering parameter; beta=1 corresponds to full posterior.
    """
    nuts_kernel = NUTS(
        lambda y_obs: _bayesian_model(
            y_obs,
            analysis_data.psd,
            starccato_model,
            rng,
            analysis_data.fmask,
            analysis_data.df,
        )
    )
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )
    mcmc.run(rng, y_obs=analysis_data.freqseries_data)
    return mcmc


def main():
    analysis_data = generate_analysis_data()
    starccato_model = StarccatoVAE()
    rng = random.PRNGKey(0)

    sig = starccato_model.generate(n=1, rng=rng)[0]
    hc = jnp.fft.rfft(sig)[analysis_data.fmask]
    # match_filter_snr(hc, analysis_data.freqseries_data, analysis_data.psd, analysis_data.df)

    mcmc = run_mcmc(analysis_data, starccato_model, 10000, 20000, 2, rng, True)
    mcmc.print_summary()
    inf_object = az.from_numpyro(mcmc)


if __name__ == "__main__":
    main()

#
#
# from starccato_sampler.post_proc import _post_process
#
#
# def main():
#     ntemps = [8, 16, 32, 64]
#
#     analysis_data = generate_analysis_data()
#     starccato_model = StarccatoVAE()
#
#     for ntemp in ntemps:
#
#         rng = random.PRNGKey(0)
#
#         sig = starccato_model.generate(n=1, rng=rng)[0]
#         hc = jnp.fft.rfft(sig)[analysis_data.fmask]
#         # match_filter_snr(hc, analysis_data.freqseries_data, analysis_data.psd, analysis_data.df)
#
#         mcmc = run_mcmc(analysis_data, starccato_model, 10000, 20000, 2, rng, True)
#         mcmc.print_summary()
#         inf_object = az.from_numpyro(mcmc)
#
#         _post_process(inf_object, analysis_data.timeseries_data, analysis_data.true_timeseries, starccato_model, mcmc,
#                       f"outdir_n{ntemp}", True, num_temps=ntemp)
