"""
UVDataset — container for interferometric visibility data.

Wraps u, v coordinates, complex visibilities, weights, and frequency axis
with convenience constructors for xradio and raw arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class UVDataset:
    """
    Container for interferometric visibility data.

    Attributes
    ----------
    u : ndarray, shape (n_baseline,)
        u-coordinates in wavelengths (λ).
    v : ndarray, shape (n_baseline,)
        v-coordinates in wavelengths (λ).
    vis_data : ndarray, shape (n_baseline, n_chan), complex
        Observed visibilities.
    weights : ndarray, shape (n_baseline, n_chan)
        Statistical weights (1/σ²).
    freqs : ndarray, shape (n_chan,)
        Channel frequencies in Hz.
    """

    u: np.ndarray
    v: np.ndarray
    vis_data: np.ndarray
    weights: np.ndarray
    freqs: np.ndarray

    def __post_init__(self) -> None:
        self.u = np.asarray(self.u, dtype=np.float64)
        self.v = np.asarray(self.v, dtype=np.float64)
        self.vis_data = np.asarray(self.vis_data, dtype=np.complex128)
        self.weights = np.asarray(self.weights, dtype=np.float64)
        self.freqs = np.asarray(self.freqs, dtype=np.float64)

        # Basic shape validation
        n_bl = self.u.shape[0]
        n_ch = self.freqs.shape[0]
        if self.v.shape[0] != n_bl:
            raise ValueError(
                f"u and v must have same length, got {n_bl} and {self.v.shape[0]}"
            )
        if self.vis_data.shape != (n_bl, n_ch):
            raise ValueError(
                f"vis_data shape {self.vis_data.shape} does not match "
                f"(n_baseline={n_bl}, n_chan={n_ch})"
            )
        if self.weights.shape != (n_bl, n_ch):
            raise ValueError(
                f"weights shape {self.weights.shape} does not match "
                f"(n_baseline={n_bl}, n_chan={n_ch})"
            )

    @property
    def n_baseline(self) -> int:
        return self.u.shape[0]

    @property
    def n_chan(self) -> int:
        return self.freqs.shape[0]

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_mock(
        cls,
        u: np.ndarray,
        v: np.ndarray,
        vis: np.ndarray,
        weights: np.ndarray,
        freqs: np.ndarray,
    ) -> UVDataset:
        """
        Construct from raw NumPy arrays (for testing / mock data).

        Parameters
        ----------
        u, v : array-like, shape (n_baseline,)
        vis : array-like, shape (n_baseline, n_chan), complex
        weights : array-like, shape (n_baseline, n_chan)
        freqs : array-like, shape (n_chan,)
        """
        return cls(u=u, v=v, vis_data=vis, weights=weights, freqs=freqs)

    @classmethod
    def from_xradio(cls, xds) -> UVDataset:
        """
        Construct from an xradio-compatible xarray Dataset.

        Expects the MSv4 schema with data variables:
        - VISIBILITY (complex): shape (..., baseline, chan, pol)
        - UVW: shape (..., baseline, 3) in metres
        - WEIGHT: shape (..., baseline, chan, pol)

        Parameters
        ----------
        xds : xarray.Dataset
            An xradio MSv4-schema Dataset.

        Returns
        -------
        UVDataset

        Notes
        -----
        This is a thin adapter. For production use, the user should
        pre-select the desired spectral window and polarization.
        """
        try:
            # Extract UVW coordinates (metres)
            uvw = xds["UVW"].values  # (time, baseline, 3)
            if uvw.ndim == 3:
                uvw = uvw.reshape(-1, 3)  # flatten time axis
            u_m = uvw[:, 0]
            v_m = uvw[:, 1]

            # Frequencies
            freqs = xds.coords["frequency"].values  # Hz

            # Convert u, v from metres to wavelengths at reference freq
            import astropy.constants as const

            ref_freq = freqs[np.size(freqs) // 2]
            wavelength = const.c.value / ref_freq
            u_lam = u_m / wavelength
            v_lam = v_m / wavelength

            # Visibilities — take first polarization
            vis = xds["VISIBILITY"].values
            while vis.ndim > 2:
                if vis.shape[-1] <= 4:
                    vis = vis[..., 0]
                else:
                    vis = vis[0]

            # Weights
            wgt = xds["WEIGHT"].values
            while wgt.ndim > 2:
                if wgt.shape[-1] <= 4:
                    wgt = wgt[..., 0]
                else:
                    wgt = wgt[0]

            return cls(u=u_lam, v=v_lam, vis_data=vis, weights=wgt, freqs=freqs)

        except KeyError as e:
            raise ValueError(
                f"xarray Dataset does not contain expected MSv4 variable: {e}"
            ) from e
