#!/usr/bin/env python
"""
Generate mock 3D visibility data for uvfit testing.

Creates:
1. A small 3D Gaussian emission cube (32×32 spatial, 16 channels)
2. Corresponding "observed" visibilities at random (u, v) points
3. Saves as NumPy .npz for easy loading in tests

The mock source is a spatially and spectrally Gaussian emission line,
centered at the image center with known parameters.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_gaussian_cube(
    nx: int = 32,
    ny: int = 32,
    n_chan: int = 16,
    cell_size: float = 0.5,
    x0: float = 0.0,
    y0: float = 0.0,
    fwhm_spatial: float = 2.0,
    fwhm_spectral: float = 4.0,
    v0_chan: float = 8.0,
    peak_flux: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """
    Create a 3D Gaussian emission cube.

    Parameters
    ----------
    nx, ny : int
        Spatial dimensions in pixels.
    n_chan : int
        Number of spectral channels.
    cell_size : float
        Pixel size in arcseconds.
    x0, y0 : float
        Source center offset from image center (arcseconds).
    fwhm_spatial : float
        Spatial FWHM in arcseconds.
    fwhm_spectral : float
        Spectral FWHM in channels.
    v0_chan : float
        Central channel of the emission.
    peak_flux : float
        Peak flux density.

    Returns
    -------
    cube : ndarray, shape (n_chan, ny, nx)
    truth : dict of true parameters
    """
    sigma_spatial = fwhm_spatial / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_spectral = fwhm_spectral / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    # Spatial grid in arcseconds, centered on image center
    x = (np.arange(nx) - nx / 2 + 0.5) * cell_size
    y = (np.arange(ny) - ny / 2 + 0.5) * cell_size
    xx, yy = np.meshgrid(x, y)

    # Channel grid
    channels = np.arange(n_chan, dtype=np.float64)

    # 3D Gaussian
    cube = np.zeros((n_chan, ny, nx), dtype=np.float64)
    spatial = peak_flux * np.exp(
        -((xx - x0) ** 2 + (yy - y0) ** 2) / (2.0 * sigma_spatial**2)
    )
    for ch in range(n_chan):
        spectral = np.exp(-((ch - v0_chan) ** 2) / (2.0 * sigma_spectral**2))
        cube[ch] = spatial * spectral

    truth = {
        "x0": x0,
        "y0": y0,
        "fwhm_spatial": fwhm_spatial,
        "fwhm_spectral": fwhm_spectral,
        "v0_chan": v0_chan,
        "peak_flux": peak_flux,
        "cell_size": cell_size,
        "nx": nx,
        "ny": ny,
        "n_chan": n_chan,
    }

    return cube, truth


def simulate_visibilities(
    cube: np.ndarray,
    cell_size: float,
    n_baselines: int = 200,
    noise_sigma: float = 0.01,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate visibility observations of a cube.

    Performs FFT of each channel, then samples at random (u, v) points
    with added Gaussian noise.

    Parameters
    ----------
    cube : ndarray, shape (n_chan, ny, nx)
    cell_size : float, arcseconds
    n_baselines : int
    noise_sigma : float
    seed : int

    Returns
    -------
    u, v : ndarray, shape (n_baselines,)
        uv coordinates in wavelengths.
    vis : ndarray, shape (n_baselines, n_chan), complex
    weights : ndarray, shape (n_baselines, n_chan)
    freqs : ndarray, shape (n_chan,)
    """
    rng = np.random.default_rng(seed)
    n_chan, ny, nx = cube.shape

    cell_rad = cell_size * np.pi / (180.0 * 3600.0)

    # uv grid coordinates
    u_grid = np.fft.fftshift(np.fft.fftfreq(nx, d=cell_rad))
    v_grid = np.fft.fftshift(np.fft.fftfreq(ny, d=cell_rad))

    # Random uv points within the grid coverage
    u_max = u_grid.max() * 0.8  # stay within 80% of max to avoid edge effects
    v_max = v_grid.max() * 0.8

    u = rng.uniform(-u_max, u_max, size=n_baselines)
    v = rng.uniform(-v_max, v_max, size=n_baselines)

    # FFT each channel and interpolate
    from scipy.interpolate import RegularGridInterpolator

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

        points = np.stack([v, u], axis=-1)
        vis[:, ch] = interp_re(points) + 1j * interp_im(points)

    # Add noise
    noise = noise_sigma * (
        rng.standard_normal(vis.shape) + 1j * rng.standard_normal(vis.shape)
    )
    vis_noisy = vis + noise

    # Weights = 1/σ²
    weights = np.full_like(vis_noisy.real, 1.0 / noise_sigma**2)

    # Dummy frequencies (channel spacing ~ 1 MHz centered at 230 GHz for CO)
    freq_center = 230.538e9  # CO(2-1) rest frequency
    chan_width = 1e6  # 1 MHz channels
    freqs = freq_center + (np.arange(n_chan) - n_chan / 2) * chan_width

    return u, v, vis_noisy, weights, freqs


def main():
    parser = argparse.ArgumentParser(description="Generate mock uvfit data")
    parser.add_argument(
        "-o", "--output", default="tests/data/mock_data.npz",
        help="Output file path (default: tests/data/mock_data.npz)",
    )
    parser.add_argument("--nx", type=int, default=32, help="Spatial pixels X")
    parser.add_argument("--ny", type=int, default=32, help="Spatial pixels Y")
    parser.add_argument("--nchan", type=int, default=16, help="Spectral channels")
    parser.add_argument("--nbl", type=int, default=200, help="Number of baselines")
    parser.add_argument("--noise", type=float, default=0.01, help="Noise sigma")
    args = parser.parse_args()

    print(f"Generating mock cube: {args.nx}×{args.ny}×{args.nchan}")
    cube, truth = make_gaussian_cube(nx=args.nx, ny=args.ny, n_chan=args.nchan)

    print(f"Simulating {args.nbl} baselines with noise σ={args.noise}")
    u, v, vis, weights, freqs = simulate_visibilities(
        cube, truth["cell_size"], n_baselines=args.nbl, noise_sigma=args.noise,
    )

    outpath = Path(args.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        outpath,
        cube=cube,
        u=u, v=v, vis=vis, weights=weights, freqs=freqs,
        **{f"truth_{k}": v for k, v in truth.items()},
    )
    print(f"Saved to {outpath}")
    print(f"  Cube shape: {cube.shape}")
    print(f"  Vis shape: {vis.shape}")
    print(f"  True params: {truth}")


if __name__ == "__main__":
    main()
