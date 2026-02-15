"""
Shared pytest fixtures for uvfit tests.
"""

from __future__ import annotations

import numpy as np
import pytest

# Add project to path
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uvfit.uvdataset import UVDataset


def make_mock_cube(
    nx: int = 32,
    ny: int = 32,
    n_chan: int = 16,
    cell_size: float = 0.5,
    x0: float = 0.0,
    y0: float = 0.0,
    peak_flux: float = 1.0,
    fwhm_spatial: float = 2.0,
    fwhm_spectral: float = 4.0,
    v0_chan: float = 8.0,
) -> np.ndarray:
    """Create a 3D Gaussian cube for testing."""
    sigma_s = fwhm_spatial / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_v = fwhm_spectral / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    x = (np.arange(nx) - nx / 2 + 0.5) * cell_size
    y = (np.arange(ny) - ny / 2 + 0.5) * cell_size
    xx, yy = np.meshgrid(x, y)

    cube = np.zeros((n_chan, ny, nx))
    spatial = peak_flux * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma_s**2))
    for ch in range(n_chan):
        cube[ch] = spatial * np.exp(-((ch - v0_chan) ** 2) / (2 * sigma_v**2))

    return cube


def make_mock_uvdata(
    cube: np.ndarray,
    cell_size: float = 0.5,
    n_baselines: int = 200,
    noise_sigma: float = 0.01,
    seed: int = 42,
) -> UVDataset:
    """Create mock UVDataset from a cube."""
    from scipy.interpolate import RegularGridInterpolator

    rng = np.random.default_rng(seed)
    n_chan, ny, nx = cube.shape
    cell_rad = cell_size * np.pi / (180.0 * 3600.0)

    u_grid = np.fft.fftshift(np.fft.fftfreq(nx, d=cell_rad))
    v_grid = np.fft.fftshift(np.fft.fftfreq(ny, d=cell_rad))

    u = rng.uniform(-u_grid.max() * 0.8, u_grid.max() * 0.8, n_baselines)
    v = rng.uniform(-v_grid.max() * 0.8, v_grid.max() * 0.8, n_baselines)

    vis = np.zeros((n_baselines, n_chan), dtype=np.complex128)
    for ch in range(n_chan):
        ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(cube[ch])))
        interp_re = RegularGridInterpolator(
            (v_grid, u_grid), ft.real, method="linear",
            bounds_error=False, fill_value=0.0,
        )
        interp_im = RegularGridInterpolator(
            (v_grid, u_grid), ft.imag, method="linear",
            bounds_error=False, fill_value=0.0,
        )
        pts = np.stack([v, u], axis=-1)
        vis[:, ch] = interp_re(pts) + 1j * interp_im(pts)

    noise = noise_sigma * (
        rng.standard_normal(vis.shape) + 1j * rng.standard_normal(vis.shape)
    )
    vis_noisy = vis + noise
    weights = np.full_like(vis_noisy.real, 1.0 / noise_sigma**2)

    freq_center = 230.538e9
    freqs = freq_center + (np.arange(n_chan) - n_chan / 2) * 1e6

    return UVDataset.from_mock(u=u, v=v, vis=vis_noisy, weights=weights, freqs=freqs)


@pytest.fixture
def cell_size():
    return 0.5


@pytest.fixture
def mock_cube(cell_size):
    """A simple 3D Gaussian cube centered at origin."""
    return make_mock_cube(cell_size=cell_size)


@pytest.fixture
def mock_uvdata(mock_cube, cell_size):
    """Mock visibility data from the Gaussian cube."""
    return make_mock_uvdata(mock_cube, cell_size=cell_size)


@pytest.fixture
def truth_params():
    """Ground-truth parameters for the mock data."""
    return {
        "x0": 0.0,
        "y0": 0.0,
        "fwhm_spatial": 2.0,
        "fwhm_spectral": 4.0,
        "v0_chan": 8.0,
        "peak_flux": 1.0,
    }
