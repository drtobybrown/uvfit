"""
NUFFT / degridding engine for uvfit.

Transforms a 3D image cube into model visibilities at arbitrary (u, v)
coordinates using per-channel 2D FFT + bilinear interpolation.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def _arcsec_to_rad(arcsec: float) -> float:
    """Convert arcseconds to radians."""
    return arcsec * np.pi / (180.0 * 3600.0)


class NUFFTEngine:
    """
    Degrid a 3D cube to model visibilities at specified (u, v) points.

    Parameters
    ----------
    cell_size : float
        Pixel size in arcseconds.
    """

    def __init__(self, cell_size: float):
        self._cell_size = float(cell_size)

    def degrid(
        self,
        cube: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        freqs: np.ndarray,
        phase_shift_arcsec: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """
        Compute model visibilities from a cube at specified (u, v) points.

        Sub-pixel spatial shifts (dx, dy) in arcsec can be applied via a
        Fourier phase ramp: V'(u,v) = V(u,v) * exp(-2πi (u·dx + v·dy)) in
        radians, preserving high-frequency information (super-resolution).

        Parameters
        ----------
        cube : ndarray, shape (n_chan, ny, nx)
            Model image cube (flux per pixel), centered at origin.
        u : ndarray, shape (n_baseline,)
            u-coordinates in wavelengths.
        v : ndarray, shape (n_baseline,)
            v-coordinates in wavelengths.
        freqs : ndarray, shape (n_chan,)
            Channel frequencies in Hz (used only for metadata; the cube
            already encodes the spectral structure).
        phase_shift_arcsec : tuple (dx, dy), optional
            Spatial shift in arcsec. If given, applied as phase ramp in
            Fourier space (no image-space interpolation).

        Returns
        -------
        model_vis : ndarray, shape (n_baseline, n_chan), complex
            Predicted visibilities.
        """
        n_chan, ny, nx = cube.shape
        n_bl = u.shape[0]

        cdtype = np.complex64 if cube.dtype == np.float32 else np.complex128

        cell_rad = self._cell_size * np.pi / (180.0 * 3600.0)
        u_grid = np.fft.fftshift(np.fft.fftfreq(nx, d=cell_rad))
        v_grid = np.fft.fftshift(np.fft.fftfreq(ny, d=cell_rad))

        model_vis = np.zeros((n_bl, n_chan), dtype=cdtype)

        for ch in range(n_chan):
            ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(cube[ch])))
            interp_re = RegularGridInterpolator(
                (v_grid, u_grid), ft.real,
                method="linear", bounds_error=False, fill_value=0.0,
            )
            interp_im = RegularGridInterpolator(
                (v_grid, u_grid), ft.imag,
                method="linear", bounds_error=False, fill_value=0.0,
            )
            points = np.stack([v, u], axis=-1)
            model_vis[:, ch] = interp_re(points) + 1j * interp_im(points)

        if phase_shift_arcsec is not None:
            dx_a, dy_a = phase_shift_arcsec
            dx_rad = _arcsec_to_rad(dx_a)
            dy_rad = _arcsec_to_rad(dy_a)
            phase = -2.0 * np.pi * (u * dx_rad + v * dy_rad)
            model_vis *= np.exp(1j * phase)[:, np.newaxis]

        return model_vis
