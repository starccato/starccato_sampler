"""
Since the VAE posteriors are all normal distributions, we can compute an analytical estimate of the evidence.

Analytical Marginal Likelihood
------------------------------

Let:
  - d = number of dimensions
  - θ = (θ₁, θ₂, …, θ_d) is the parameter vector

Priors:
  For each i = 1, …, d:
    p(θᵢ) = 𝒩(0, 1)
          = 1/√(2π) * exp( - θᵢ² / 2 )

Posterior:
  For each i = 1, …, d:
    p(θᵢ | D) = 𝒩(μᵢ, σᵢ²)

Marginal Likelihood (Evidence):
  p(D) = p(D | θ) * p(θ) / p(θ | D)


Using the Laplace approximation, where we assume that the posterior
is approximately Gaussian around its mode, we
evaluate at θ = θr, the posterior mean.


1. Prior at θ = θr:
     p(θr) = ∏[i=1 to d] (1/√(2π) * exp( - θrᵢ² / 2 ))
          = (2π)^(-d/2) * exp( - (1/2) * Σ[i=1 to d] θrᵢ² )

2. Posterior at θ = θr:
     p(θr | D) = ∏[i=1 to d] (1 / (√(2π) * σᵢ))
              = (2π)^(-d/2) * ∏[i=1 to d] (1/σᵢ)

3. Combining to Get p(D):
     p(D) = p(D | θr) * (p(θr) / p(θr | D))
          = p(D | θr) * { (2π)^(-d/2) * exp( - (1/2) * Σ[i=1 to d] θrᵢ² ) }
                      / { (2π)^(-d/2) * ∏[i=1 to d] (1/σᵢ) }
          = p(D | θr) * exp( - (1/2) * Σ[i=1 to d] θrᵢ² ) * ∏[i=1 to d] σᵢ

Log-Domain Expression:
     log p(D) = log p(D | θr)
              - (1/2) * Σ[i=1 to d] θrᵢ²
              + Σ[i=1 to d] log(σᵢ)
"""

import numpy as np


def compute_gaussian_approx_evidence(
    ref_sample: np.ndarray,
    lnl_ref: float,
    posterior_samples: np.ndarray,
    n_bootstraps: int = 10,
    bootstrap_frac: float = 0.8,
):
    """
    Compute the Gaussian Approximation to the log marginal likelihood (ln Z)
    using the MAP estimate.

    Parameters:
    - LnL_map : float
         Log-likelihood evaluated at the MAP estimate.
    - ref_sample : array-like of shape (d,)
         The MAP estimate of the parameters.
    - posterior_samples : np.ndarray of shape (n_samples, d)
         Posterior samples used to compute the standard deviations.

    Returns:
    - lnZ : float
         The estimated log marginal likelihood.

    Formula:
      lnZ = LnL_map - 0.5 * sum(theta_map^2) + sum(log(sigma)),
      where sigma is computed across the posterior samples.
    """

    nsamp, ndim = posterior_samples.shape
    assert ref_sample.shape == (
        ndim,
    ), f"Shape mismatch between ref_sample ({ref_sample.shape}) and posterior_samples dim ({ndim, })"

    args = (ref_sample, lnl_ref, posterior_samples, bootstrap_frac)
    lnz = np.array([_compute_lnz(*args) for _ in range(n_bootstraps)])
    return np.mean(lnz), np.std(lnz)


def _compute_lnz(
    ref_sample: np.ndarray,
    lnl_ref: float,
    posterior_samples: np.ndarray,
    frac: float,
) -> float:
    nsamp, ndim = posterior_samples.shape
    idxs = np.random.choice(
        np.arange(nsamp), size=int(frac * nsamp), replace=False
    )
    samples = posterior_samples[idxs]

    # Compute standard deviation for each parameter (sigma)
    sigma = np.std(samples, axis=0)
    assert (
        sigma.shape == ref_sample.shape
    ), f"Shape mismatch between sigma ({sigma.shape} and ref_sample {ref_sample.shape})"

    # Compute the quadratic penalty term (0.5 * sum(theta_map^2))
    penalty = 0.5 * np.sum(np.square(ref_sample))

    # Compute the "volume" term (sum(log(sigma)))
    sigma_term = np.sum(np.log(sigma))

    # Combine to get the Gaussian approximation to ln Z
    lnZ = lnl_ref - penalty + sigma_term
    return lnZ
