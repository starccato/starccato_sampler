"""
pip install numpyro jaxns arviz

"""

import os

import arviz as az
import jax.numpy as jnp
import jax.random
import numpy as np
import numpyro
from jax.random import PRNGKey
from numpyro.contrib.nested_sampling import NestedSampler
from starccato_jax import StarccatoVAE
from tqdm.auto import tqdm

from starccato_sampler.core import _bayesian_model
from starccato_sampler.sampler import sample

numpyro.enable_x64(False)

noise_sigma = 1.0
starccato_model = StarccatoVAE()
rng = PRNGKey(0)
beta = 1.0

z_true = jnp.array(np.random.normal(0, 1, (1, starccato_model.latent_dim)))
y_true = starccato_model.generate(z=z_true, rng=rng)[0]
y_obs = y_true + jax.random.normal(rng, y_true.shape) * noise_sigma

OUT = "output2"
os.makedirs(OUT, exist_ok=True)

RESULT_FILE = f"{OUT}/lnz_results.txt"

# write header
with open(RESULT_FILE, "w") as f:
    f.write("NS_LnZ SS_LnZ\n")


def run_analysis(i):
    rng = PRNGKey(i)
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
