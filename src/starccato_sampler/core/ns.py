
from numpyro.infer import MCMC, NUTS
import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from jax.random import PRNGKey

from starccato_jax.starccato_model import StarccatoModel
from jaxtyping import Array
from numpyro.contrib.nested_sampling import NestedSampler
from .bayesian_functions import _bayesian_model


def _run_ns(
    y_obs: jnp.ndarray,
    starccato_model: StarccatoModel,
    num_samples: int,
    num_warmup: int,
    num_chains: int,
    rng: PRNGKey,
    beta: float = 1.0,
    progress_bar: bool = False,
    noise_sigma: float = 1.0,
    reference_prior: Array = None
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

    rng = random.PRNGKey(rng_int)
    ns = NestedSampler(
        _bayesian_model, constructor_kwargs=dict(max_samples=5000)
    )
    ns.run(
        rng,
        y_obs=y_obs,
        starccato_model=starccato_model,
        beta=beta,
        noise_sigma=noise_sigma,
        rng=rng,
    )
    ns.print_summary()
    ns_lnz = ns._results.log_Z_mean
    # TODO: save as arviz inference object --> netcdf
    # all our postprocessing is done on the netcdf file

    mcmc_out = f"{OUT}/mcmc_{i}"
    os.makedirs(mcmc_out, exist_ok=True)
    sample(
        data=y_obs,
        starccato_model=starccato_model,
        rng_int=i,
        outdir=mcmc_out,
        stepping_stone_lnz=True,
        noise_sigma=noise_sigma,
    )
    inf_obj = az.from_netcdf(f"{mcmc_out}/inference.nc")
    ss_lnz = float(inf_obj.sample_stats["ss_lnz"])

    # append the results to a file
    with open(RESULT_FILE, "a") as f:
        f.write(f"{ns_lnz} {ss_lnz}\n")


for i in tqdm(range(100)):
    run_analysis(i)

