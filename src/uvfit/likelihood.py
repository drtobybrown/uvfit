"""
Visibility-space likelihood for uvfit.

Computes χ² = Σ wᵢ |V_obs,i − V_mod,i|² over all baselines and channels.
This is the core objective function for fitting.
"""

from __future__ import annotations

import numpy as np


class VisibilityLikelihood:
    """
    Visibility-space chi-squared likelihood.

    The statistic is:

        χ² = Σᵢ (wᵢ · s) |V_obs,i − V_mod,i|²

    where the sum runs over all baselines × channels, wᵢ = 1/σᵢ²,
    s is the ``weight_scale_factor`` (default 1.0; use 0.5 for
    Hanning-smoothed ALMA data to account for spectral covariance),
    and |·|² denotes the squared complex modulus (real² + imag²).

    Parameters
    ----------
    weight_scale_factor : float
        Multiplicative correction applied to CASA-exported visibility
        weights before computing χ².  For Hanning-smoothed ALMA data,
        adjacent channels have ~50 % correlated noise, so the effective
        independent DoF is roughly halved; set to 0.5 in that case.
        Default: 1.0 (no correction).
    """

    def __init__(self, weight_scale_factor: float = 1.0) -> None:
        if weight_scale_factor <= 0.0:
            raise ValueError(
                f"weight_scale_factor must be positive; got {weight_scale_factor}"
            )
        self._weight_scale_factor = float(weight_scale_factor)

    @property
    def weight_scale_factor(self) -> float:
        """Current weight scaling factor."""
        return self._weight_scale_factor

    def chi_squared(
        self,
        model_vis: np.ndarray,
        observed_vis: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """
        Compute the weighted chi-squared.

        Parameters
        ----------
        model_vis : ndarray, shape (n_baseline, n_chan), complex
        observed_vis : ndarray, shape (n_baseline, n_chan), complex
        weights : ndarray, shape (n_baseline, n_chan)

        Returns
        -------
        chi2 : float
        """
        residual = observed_vis - model_vis
        scaled_weights = weights * self._weight_scale_factor
        return float(np.sum(scaled_weights * np.abs(residual) ** 2))

    def reduced_chi_squared(
        self,
        model_vis: np.ndarray,
        observed_vis: np.ndarray,
        weights: np.ndarray,
        n_params: int,
    ) -> float:
        """
        Compute the reduced chi-squared (χ²/ν).

        Parameters
        ----------
        n_params : int
            Number of free model parameters.

        Returns
        -------
        chi2_red : float
        """
        chi2 = self.chi_squared(model_vis, observed_vis, weights)
        n_data = 2 * observed_vis.size  # real + imag degrees of freedom
        dof = max(n_data - n_params, 1)
        return chi2 / dof

    def log_likelihood(
        self,
        model_vis: np.ndarray,
        observed_vis: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """
        Log-likelihood (up to a constant).

        ln L = -0.5 * χ²

        This is the function to maximize in MCMC / Bayesian fitting.
        """
        return -0.5 * self.chi_squared(model_vis, observed_vis, weights)

    def __call__(
        self,
        model_vis: np.ndarray,
        observed_vis: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """Alias for chi_squared."""
        return self.chi_squared(model_vis, observed_vis, weights)
