"""
Tests for the VisibilityLikelihood class.
"""

from __future__ import annotations

import numpy as np
import pytest

from uvfit.likelihood import VisibilityLikelihood


class TestVisibilityLikelihood:
    """Tests for χ² computation."""

    def setup_method(self):
        self.likelihood = VisibilityLikelihood()

    def test_zero_residual(self):
        """Identical model and observed → χ² = 0."""
        vis = np.ones((10, 4), dtype=np.complex128)
        weights = np.ones((10, 4))

        chi2 = self.likelihood.chi_squared(vis, vis, weights)
        assert chi2 == pytest.approx(0.0, abs=1e-15)

    def test_known_residual(self):
        """
        Known residual should give a predictable χ².
        If residual = (1 + 0j) everywhere, and weights = 1:
        χ² = Σ w |r|² = N * 1 * 1 = N
        """
        n_bl, n_chan = 10, 4
        obs = np.ones((n_bl, n_chan), dtype=np.complex128)
        model = np.zeros((n_bl, n_chan), dtype=np.complex128)
        weights = np.ones((n_bl, n_chan))

        chi2 = self.likelihood.chi_squared(model, obs, weights)
        assert chi2 == pytest.approx(n_bl * n_chan, rel=1e-10)

    def test_weights_matter(self):
        """Doubling weights should double χ²."""
        obs = np.ones((5, 2), dtype=np.complex128) * (1 + 1j)
        model = np.zeros((5, 2), dtype=np.complex128)
        w1 = np.ones((5, 2))
        w2 = np.ones((5, 2)) * 2.0

        chi2_1 = self.likelihood.chi_squared(model, obs, w1)
        chi2_2 = self.likelihood.chi_squared(model, obs, w2)

        assert chi2_2 == pytest.approx(2.0 * chi2_1, rel=1e-10)

    def test_reduced_chi_squared(self):
        """Reduced χ² = χ² / dof."""
        obs = np.ones((10, 4), dtype=np.complex128)
        model = np.zeros((10, 4), dtype=np.complex128)
        weights = np.ones((10, 4))

        n_params = 3
        red_chi2 = self.likelihood.reduced_chi_squared(
            model, obs, weights, n_params
        )

        chi2 = self.likelihood.chi_squared(model, obs, weights)
        n_data = 2 * obs.size  # real + imag
        dof = n_data - n_params
        expected = chi2 / dof

        assert red_chi2 == pytest.approx(expected, rel=1e-10)

    def test_log_likelihood(self):
        """log L = -0.5 * χ²."""
        obs = np.ones((5, 3), dtype=np.complex128) * 2.0
        model = np.ones((5, 3), dtype=np.complex128)
        weights = np.ones((5, 3))

        chi2 = self.likelihood.chi_squared(model, obs, weights)
        log_l = self.likelihood.log_likelihood(model, obs, weights)

        assert log_l == pytest.approx(-0.5 * chi2, rel=1e-10)

    def test_callable(self):
        """__call__ should match chi_squared."""
        obs = np.ones((5, 3), dtype=np.complex128)
        model = np.zeros((5, 3), dtype=np.complex128)
        weights = np.ones((5, 3))

        assert self.likelihood(model, obs, weights) == pytest.approx(
            self.likelihood.chi_squared(model, obs, weights)
        )

    def test_complex_residual(self):
        """
        Complex residual: |r|² = Re(r)² + Im(r)².
        If r = 3 + 4j, |r|² = 25.
        """
        obs = np.array([[3.0 + 4.0j]], dtype=np.complex128)
        model = np.array([[0.0 + 0.0j]], dtype=np.complex128)
        weights = np.array([[1.0]])

        chi2 = self.likelihood.chi_squared(model, obs, weights)
        assert chi2 == pytest.approx(25.0, rel=1e-10)
