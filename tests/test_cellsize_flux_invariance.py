"""
Sub-Agent 1 acceptance test: Cell-size FFT flux invariance.

Two identical Gaussian sources with different cellsize (0.1" and 0.3")
must produce identical total visibility flux after degridding.
"""

from __future__ import annotations

import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uvfit.nufft import NUFFTEngine


def _make_gaussian_cube(
    nx: int,
    ny: int,
    n_chan: int,
    cell_size: float,
    fwhm_arcsec: float = 2.0,
    peak_flux: float = 1.0,
) -> np.ndarray:
    """Gaussian source cube for a given pixel grid."""
    sigma = fwhm_arcsec / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    x = (np.arange(nx) - nx / 2 + 0.5) * cell_size
    y = (np.arange(ny) - ny / 2 + 0.5) * cell_size
    xx, yy = np.meshgrid(x, y)
    image = peak_flux * np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return np.stack([image] * n_chan)


class TestCellSizeFluxInvariance:
    """The degridded zero-spacing flux must be identical for different cell sizes."""

    def test_total_flux_invariant_to_cellsize(self):
        """
        A Gaussian source degridded at u=0, v=0 (zero spacing = total flux)
        must return the same amplitude regardless of cellsize_arcsec.
        """
        n_chan = 2
        fwhm = 2.0  # arcsec

        # Two grids: fine (0.1") and coarse (0.3")
        configs = [
            {"cell": 0.1, "nx": 128, "ny": 128},
            {"cell": 0.3, "nx": 48, "ny": 48},
        ]

        # Sample at zero spacing (total flux) and a few other points
        u = np.array([0.0, 100.0, 500.0])
        v = np.array([0.0, 0.0, 0.0])
        freqs = np.array([230.538e9, 231.0e9])

        fluxes = []
        for cfg in configs:
            cube = _make_gaussian_cube(
                cfg["nx"], cfg["ny"], n_chan,
                cell_size=cfg["cell"],
                fwhm_arcsec=fwhm,
                peak_flux=1.0,
            )
            engine = NUFFTEngine(cell_size=cfg["cell"])
            vis = engine.degrid(cube, u, v, freqs)

            # Zero-spacing visibility = total integrated flux (Jy·sr)
            total_flux = float(np.abs(vis[0, 0]))
            fluxes.append(total_flux)

        # The two total fluxes must match within 2%
        assert fluxes[0] > 0, "Zero flux from fine grid"
        assert fluxes[1] > 0, "Zero flux from coarse grid"
        np.testing.assert_allclose(
            fluxes[0], fluxes[1], rtol=0.02,
            err_msg=(
                f"Total flux varies with cell size: "
                f"cell=0.1\" → {fluxes[0]:.6e}, cell=0.3\" → {fluxes[1]:.6e}"
            ),
        )

    def test_uv_amplitude_profile_invariant(self):
        """
        The visibility amplitude profile vs UV distance should be
        consistent between different cell sizes (same physical source).
        """
        n_chan = 1
        configs = [
            {"cell": 0.1, "nx": 128, "ny": 128},
            {"cell": 0.3, "nx": 48, "ny": 48},
        ]

        u = np.array([0.0, 50.0, 200.0, 1000.0])
        v = np.zeros(4)
        freqs = np.array([230.538e9])

        profiles = []
        for cfg in configs:
            cube = _make_gaussian_cube(cfg["nx"], cfg["ny"], n_chan, cfg["cell"])
            engine = NUFFTEngine(cell_size=cfg["cell"])
            vis = engine.degrid(cube, u, v, freqs)
            profiles.append(np.abs(vis[:, 0]))

        # Normalize to zero-spacing and compare shapes
        for i in range(len(profiles)):
            profiles[i] = profiles[i] / profiles[i][0]

        np.testing.assert_allclose(
            profiles[0], profiles[1], rtol=0.05,
            err_msg="Visibility profile shape varies with cell size",
        )
