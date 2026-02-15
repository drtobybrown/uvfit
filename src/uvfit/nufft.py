"""
NUFFT / degridding engine for uvfit.

Transforms a 3D image cube into model visibilities at arbitrary (u, v)
coordinates. Uses per-channel 2D transforms.

Two strategies:
- CPU fallback: FFT + bilinear interpolation in the uv-plane
- JAX: jax.numpy.fft + differentiable interpolation (or jax-finufft if available)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class NUFFTEngine:
    """
    Degrid a 3D cube to model visibilities at specified (u, v) points.

    Parameters
    ----------
    cell_size : float
        Pixel size in arcseconds.
    backend : str
        Array backend name ("numpy" or "jax"). Default: "numpy".
    """

    def __init__(self, cell_size: float, backend: str = "numpy"):
        self._cell_size = float(cell_size)
        self._backend_name = backend

    def degrid(
        self,
        cube: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        freqs: np.ndarray,
    ) -> np.ndarray:
        """
        Compute model visibilities from a cube at specified (u, v) points.

        Parameters
        ----------
        cube : ndarray, shape (n_chan, ny, nx)
            Model image cube (flux per pixel).
        u : ndarray, shape (n_baseline,)
            u-coordinates in wavelengths.
        v : ndarray, shape (n_baseline,)
            v-coordinates in wavelengths.
        freqs : ndarray, shape (n_chan,)
            Channel frequencies in Hz (used only for metadata; the cube
            already encodes the spectral structure).

        Returns
        -------
        model_vis : ndarray, shape (n_baseline, n_chan), complex
            Predicted visibilities.
        """
        if self._backend_name == "jax":
            return self._degrid_jax(cube, u, v)
        else:
            return self._degrid_numpy(cube, u, v)

    def _degrid_numpy(
        self, cube: np.ndarray, u: np.ndarray, v: np.ndarray
    ) -> np.ndarray:
        """
        CPU fallback: FFT each channel, then bilinear-interpolate at (u, v).

        The FFT gives us the visibility function on a regular grid in
        (u, v). We then interpolate to the actual baseline coordinates.
        """
        n_chan, ny, nx = cube.shape
        n_bl = u.shape[0]

        # Pixel size in radians
        cell_rad = self._cell_size * np.pi / (180.0 * 3600.0)

        # uv grid spacing: Δu = 1 / (N * Δθ)
        du = 1.0 / (nx * cell_rad)
        dv = 1.0 / (ny * cell_rad)

        # uv grid coordinates (after fftshift)
        u_grid = np.fft.fftshift(np.fft.fftfreq(nx, d=cell_rad))
        v_grid = np.fft.fftshift(np.fft.fftfreq(ny, d=cell_rad))

        model_vis = np.zeros((n_bl, n_chan), dtype=np.complex128)

        for ch in range(n_chan):
            # 2D FFT of the image (shift zero-frequency to center first)
            ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(cube[ch])))

            # Interpolate real and imaginary parts separately
            interp_re = RegularGridInterpolator(
                (v_grid, u_grid),
                ft.real,
                method="linear",
                bounds_error=False,
                fill_value=0.0,
            )
            interp_im = RegularGridInterpolator(
                (v_grid, u_grid),
                ft.imag,
                method="linear",
                bounds_error=False,
                fill_value=0.0,
            )

            # Sample at (v, u) points — note the axis ordering
            points = np.stack([v, u], axis=-1)
            model_vis[:, ch] = interp_re(points) + 1j * interp_im(points)

        return model_vis

    def _degrid_jax(
        self, cube: np.ndarray, u: np.ndarray, v: np.ndarray
    ) -> np.ndarray:
        """
        JAX path: differentiable FFT + bilinear interpolation.

        Uses jax.numpy for the FFT and a custom differentiable
        bilinear interpolation. Falls back to jax-finufft if available.
        """
        try:
            import jax
            import jax.numpy as jnp
        except ImportError:
            raise ImportError(
                "JAX backend requires jax. Install with: pip install uvfit[jax]"
            )

        cube_j = jnp.array(cube)
        u_j = jnp.array(u)
        v_j = jnp.array(v)

        cell_rad = self._cell_size * jnp.pi / (180.0 * 3600.0)
        n_chan, ny, nx = cube_j.shape

        u_grid = jnp.fft.fftshift(jnp.fft.fftfreq(nx, d=cell_rad))
        v_grid = jnp.fft.fftshift(jnp.fft.fftfreq(ny, d=cell_rad))

        def degrid_channel(image_2d):
            ft = jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.ifftshift(image_2d)))
            return self._bilinear_interp_jax(ft, u_grid, v_grid, u_j, v_j)

        model_vis = jax.vmap(degrid_channel)(cube_j)  # (n_chan, n_bl)
        return np.asarray(model_vis.T)  # (n_bl, n_chan)

    @staticmethod
    def _bilinear_interp_jax(grid, u_coords, v_coords, u_query, v_query):
        """
        Differentiable bilinear interpolation on a regular 2D grid.
        """
        import jax.numpy as jnp

        ny, nx = grid.shape

        # Convert physical coords to fractional pixel indices
        du = u_coords[1] - u_coords[0]
        dv = v_coords[1] - v_coords[0]

        x_frac = (u_query - u_coords[0]) / du
        y_frac = (v_query - v_coords[0]) / dv

        # Clamp to valid range
        x_frac = jnp.clip(x_frac, 0, nx - 1.001)
        y_frac = jnp.clip(y_frac, 0, ny - 1.001)

        x0 = jnp.floor(x_frac).astype(jnp.int32)
        y0 = jnp.floor(y_frac).astype(jnp.int32)
        x1 = x0 + 1
        y1 = y0 + 1

        x1 = jnp.minimum(x1, nx - 1)
        y1 = jnp.minimum(y1, ny - 1)

        wx = x_frac - x0
        wy = y_frac - y0

        val = (
            grid[y0, x0] * (1 - wx) * (1 - wy)
            + grid[y0, x1] * wx * (1 - wy)
            + grid[y1, x0] * (1 - wx) * wy
            + grid[y1, x1] * wx * wy
        )
        return val
