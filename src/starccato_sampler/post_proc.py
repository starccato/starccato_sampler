import numpy as np


def _calculate_coverage(self):
    spec_mat_lower_real = np.zeros_like(
        self.spectral_density_q[0], dtype=float
    )
    for j in range(self.n_freq):
        spec_mat_lower_real[j] = self._complex_to_real(
            self.spectral_density_q[0][j]
        )

    spec_mat_upper_real = np.zeros_like(
        self.spectral_density_q[2], dtype=float
    )
    for j in range(self.n_freq):
        spec_mat_upper_real[j] = self._complex_to_real(
            self.spectral_density_q[2][j]
        )
    coverage_point_CI = np.mean(
        (spec_mat_lower_real <= self.real_spec_true)
        & (self.real_spec_true <= spec_mat_upper_real)
    )
    return coverage_point_CI

    def _calculate_L2_error(self):
        spec_mat_median = self.spectral_density_q[1]
        N2_VI = np.empty(self.n_freq)
        for i in range(self.n_freq):
            N2_VI[i] = np.sum(
                np.diag(
                    (spec_mat_median[i, :, :] - self.spec_true[i, :, :])
                    @ (spec_mat_median[i, :, :] - self.spec_true[i, :, :])
                )
            )
        L2_VI = np.sqrt(np.mean(N2_VI))
        return L2_VI


def __get_uniform_ci(psd_all, psd_q):
    psd_median = psd_q[1]
    n_samples, n_freq, p, _ = psd_all.shape

    # transform elements of psd_all and psd_median to the real numbers
    real_psd_all = np.zeros_like(psd_all, dtype=float)
    real_psd_median = np.zeros_like(psd_median, dtype=float)

    for i in range(n_samples):
        for j in range(n_freq):
            real_psd_all[i, j] = __complex_to_real(psd_all[i, j])

    for j in range(n_freq):
        real_psd_median[j] = __complex_to_real(psd_median[j])

    # find the maximum normalized absolute deviation for real_psd_all
    max_std_abs_dev = __uniformmax_multi(real_psd_all)
    # find the threshold of the 90% as a screening criterion
    threshold = np.quantile(max_std_abs_dev, 0.9)

    # find the uniform CI for real_psd_median
    mad = median_abs_deviation(real_psd_all, axis=0, nan_policy="omit")
    mad[mad == 0] = 1e-10
    lower_bound = real_psd_median - threshold * mad
    upper_bound = real_psd_median + threshold * mad

    # Converts lower_bound and upper_bound to the complex matrix
    psd_uni_lower = np.zeros_like(lower_bound, dtype=complex)
    psd_uni_upper = np.zeros_like(upper_bound, dtype=complex)

    for i in range(n_freq):
        psd_uni_lower[i] = __real_to_complex(lower_bound[i])
        psd_uni_upper[i] = __real_to_complex(upper_bound[i])

    psd_uniform = np.stack([psd_uni_lower, psd_median, psd_uni_upper], axis=0)

    return psd_uniform


# Find the normalized median absolute deviation for every element among all smaples
# For all samples of each frequency, each matrix, return their maximum normalized absolute deviation
def __uniformmax_multi(mSample):
    N_sample, N, d, _ = mSample.shape
    C_help = np.zeros((N_sample, N, d, d))

    for j in range(N):
        for r in range(d):
            for s in range(d):
                C_help[:, j, r, s] = __uniformmax_help(mSample[:, j, r, s])

    return np.max(C_help, axis=0)


def __uniformmax_help(sample):
    return np.abs(sample - np.median(sample)) / median_abs_deviation(sample)


def __get_pointwise_ci(psd_all, quantiles):
    _, num_freq, p_dim, _ = psd_all.shape
    psd_q = np.zeros((3, num_freq, p_dim, p_dim), dtype=complex)

    diag_indices = np.diag_indices(p_dim)
    psd_q[:, :, diag_indices[0], diag_indices[1]] = np.quantile(
        np.real(psd_all[:, :, diag_indices[0], diag_indices[1]]),
        quantiles,
        axis=0,
    )

    # we dont do lower triangle because it is symmetric
    upper_triangle_idx = np.triu_indices(p_dim, k=1)
    real_part = np.real(
        psd_all[:, :, upper_triangle_idx[1], upper_triangle_idx[0]]
    )
    imag_part = np.imag(
        psd_all[:, :, upper_triangle_idx[1], upper_triangle_idx[0]]
    )

    for i, q in enumerate(quantiles):
        psd_q[i, :, upper_triangle_idx[1], upper_triangle_idx[0]] = (
            np.quantile(real_part, q, axis=0)
            + 1j * np.quantile(imag_part, q, axis=0)
        ).T

    psd_q[:, :, upper_triangle_idx[0], upper_triangle_idx[1]] = np.conj(
        psd_q[:, :, upper_triangle_idx[1], upper_triangle_idx[0]]
    )
    return psd_q
