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

        χ² = Σᵢ wᵢ |V_obs,i − V_mod,i|²

    where the sum runs over all baselines × channels, wᵢ = 1/σᵢ²,
    and |·|² denotes the squared complex modulus (real² + imag²).
    """

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
        return float(np.sum(weights * np.abs(residual) ** 2))

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
