import os

import arviz as az
import numpy as np
import pytest
import xarray as xr

from starccato_sampler.pp_test import pp_test


def generate_fake_results(
    outdir: str, num_results: int = 100, ndraw: int = 200, ndim: int = 4
):
    # Generate fake results
    np.random.seed(42)
    for i in range(num_results):
        _save_fake_res(f"{outdir}/fake_{i}", ndraw=ndraw, dim=ndim)


def _save_fake_res(
    outdir: str, nchain: int = 2, ndraw: int = 200, dim: int = 4
):
    fname = os.path.join(outdir, "inference.nc")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    z_post = np.random.randn(nchain, ndraw, dim)
    z_true = np.random.randn(dim)
    # Convert to xarray.DataArray
    z_post_xr = xr.DataArray(
        z_post,
        dims=["chain", "draw", "dim"],  # Define dimension names
        coords={
            "chain": np.arange(nchain),
            "draw": np.arange(ndraw),
            "dim": np.arange(dim),
        },  # Coordinate labels
        name="z",  # Give it a name
    )
    idata = az.from_dict(
        posterior={"z": z_post_xr}, sample_stats={"true_latent": z_true}
    )

    assert (
        idata.posterior["z"].stack(sample=("chain", "draw")).values.size > 0
    )  # Ensure it's not empty
    assert (
        idata.sample_stats["true_latent"].to_numpy().ravel().size > 0
    )  # Ensure it's not empty
    idata.to_netcdf(fname)


def test_pp_test(tmpdir, outdir):
    dim = 4
    generate_fake_results(tmpdir, num_results=10, ndraw=20, ndim=dim)
    pp_test(
        result_regex=f"{tmpdir}/fake_*/inference.nc",
        credible_levels_fpath=f"{tmpdir}/credible_levels.npy",
        plot_fname=f"{outdir}/pp_plot.png",
        include_title=True,
        dim=dim,
    )
