"""
Tests for the NUFFT / degridding engine.

Validates that FFT + bilinear interpolation produces visibilities
that agree with a direct DFT computation.
"""

from __future__ import annotations

import numpy as np
import pytest

from uvfit.nufft import NUFFTEngine


class TestNUFFTEngine:
    """Tests for NUFFTEngine degridding accuracy."""

    def test_point_source_at_center(self):
        """
        A point source at the image center should produce constant
        (flat) visibilities across all (u, v) points.
        """
        nx, ny, n_chan = 32, 32, 4
        cell_size = 0.5  # arcsec

        # Point source at center
        cube = np.zeros((n_chan, ny, nx))
        cube[:, ny // 2, nx // 2] = 1.0

        # Random uv points (within grid limits)
        rng = np.random.default_rng(123)
        cell_rad = cell_size * np.pi / (180.0 * 3600.0)
        pixel_solid_angle = cell_rad ** 2
        u_max = 0.4 / cell_rad  # stay well within grid
        n_bl = 50
        u = rng.uniform(-u_max, u_max, n_bl)
        v = rng.uniform(-u_max, u_max, n_bl)
        freqs = np.linspace(230e9, 231e9, n_chan)

        engine = NUFFTEngine(cell_size=cell_size)
        model_vis = engine.degrid(cube, u, v, freqs)

        # All visibilities should have approximately the same amplitude
        # (a centered point source has flat |V(u,v)| = pixel_solid_angle)
        amplitudes = np.abs(model_vis)
        assert amplitudes.shape == (n_bl, n_chan)
        # Allow some tolerance for interpolation error
        np.testing.assert_allclose(amplitudes, pixel_solid_angle, atol=0.15 * pixel_solid_angle)

    def test_degrid_vs_direct_dft(self):
        """
        Compare degridded visibilities to a direct DFT for a simple source.
        """
        nx, ny = 16, 16
        n_chan = 2
        cell_size = 1.0  # arcsec
        cell_rad = cell_size * np.pi / (180.0 * 3600.0)

        # Simple Gaussian source at center
        x = (np.arange(nx) - nx / 2 + 0.5) * cell_size
        y = (np.arange(ny) - ny / 2 + 0.5) * cell_size
        xx, yy = np.meshgrid(x, y)
        sigma = 2.0  # arcsec
        image = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

        cube = np.stack([image, image * 0.5])  # 2 channels

        # A few uv points
        u = np.array([10.0, 50.0, -30.0, 0.0])
        v = np.array([20.0, -10.0, 40.0, 0.0])
        freqs = np.array([230e9, 231e9])

        # Degrid
        engine = NUFFTEngine(cell_size=cell_size)
        model_vis = engine.degrid(cube, u, v, freqs)

        # Direct DFT for comparison (include pixel solid angle to match NUFFTEngine)
        pixel_solid_angle = cell_rad ** 2
        x_rad = (np.arange(nx) - nx / 2 + 0.5) * cell_rad
        y_rad = (np.arange(ny) - ny / 2 + 0.5) * cell_rad
        xx_rad, yy_rad = np.meshgrid(x_rad, y_rad)

        dft_vis = np.zeros((4, 2), dtype=np.complex128)
        for i in range(4):
            phase = -2.0 * np.pi * (u[i] * xx_rad + v[i] * yy_rad)
            kernel = np.exp(1j * phase)
            for ch in range(2):
                dft_vis[i, ch] = np.sum(cube[ch] * kernel) * pixel_solid_angle

        # Should agree within interpolation tolerance
        np.testing.assert_allclose(
            np.abs(model_vis), np.abs(dft_vis), rtol=0.15
        )

    def test_zero_cube_gives_zero_vis(self):
        """An empty cube should give zero visibilities."""
        cube = np.zeros((4, 16, 16))
        u = np.array([10.0, 20.0])
        v = np.array([15.0, 25.0])
        freqs = np.linspace(230e9, 231e9, 4)

        engine = NUFFTEngine(cell_size=1.0)
        model_vis = engine.degrid(cube, u, v, freqs)

        np.testing.assert_allclose(model_vis, 0.0, atol=1e-15)

    def test_output_shape(self):
        """Verify output shape matches (n_baseline, n_chan)."""
        n_bl, n_chan = 30, 8
        cube = np.random.default_rng(0).standard_normal((n_chan, 16, 16))
        u = np.random.default_rng(1).uniform(-100, 100, n_bl)
        v = np.random.default_rng(2).uniform(-100, 100, n_bl)
        freqs = np.linspace(230e9, 231e9, n_chan)

        engine = NUFFTEngine(cell_size=1.0)
        model_vis = engine.degrid(cube, u, v, freqs)

        assert model_vis.shape == (n_bl, n_chan)
        assert np.iscomplexobj(model_vis)
