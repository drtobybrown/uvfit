"""
Forward models for uvfit.

Defines the abstract base class and concrete implementations that map
a parameter vector to a 3D image cube (n_chan, ny, nx).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy.ndimage import shift as ndshift


class ForwardModel(ABC):
    """
    Abstract base class for 3D forward models.

    A ForwardModel takes a dictionary of parameters and returns a
    3D image cube of shape (n_chan, ny, nx).
    """

    @abstractmethod
    def generate_cube(self, params: dict[str, float]) -> np.ndarray:
        """
        Generate a 3D image cube from model parameters.

        Parameters
        ----------
        params : dict[str, float]
            Parameter name → value mapping.

        Returns
        -------
        cube : ndarray, shape (n_chan, ny, nx)
            The model image cube (flux density per pixel).
        """
        ...

    @property
    @abstractmethod
    def param_names(self) -> list[str]:
        """List of parameter names this model expects."""
        ...

    @property
    @abstractmethod
    def default_params(self) -> dict[str, float]:
        """Default parameter values."""
        ...

    @property
    @abstractmethod
    def bounds(self) -> dict[str, tuple[float, float]]:
        """Parameter bounds as {name: (lower, upper)}."""
        ...

    @property
    @abstractmethod
    def cell_size(self) -> float:
        """Pixel size in arcseconds."""
        ...


class TemplateCubeModel(ForwardModel):
    """
    Forward model that shifts/scales a reference template cube.

    Parameters
    ----------
    template_cube : ndarray, shape (n_chan, ny, nx)
        The reference 3D image cube to modify.
    cell_size_arcsec : float
        Pixel size in arcseconds.
    channel_width_kms : float, optional
        Channel width in km/s for velocity shifts. Default: 1.0.

    Model Parameters
    ----------------
    dx : float — spatial shift in x (arcseconds)
    dy : float — spatial shift in y (arcseconds)
    dv : float — velocity shift in channels (fractional)
    flux_scale : float — multiplicative flux scaling
    """

    def __init__(
        self,
        template_cube: np.ndarray,
        cell_size_arcsec: float,
        channel_width_kms: float = 1.0,
    ):
        self._template = np.array(template_cube, dtype=np.float64)
        self._cell_size = float(cell_size_arcsec)
        self._channel_width = float(channel_width_kms)

        if self._template.ndim != 3:
            raise ValueError(
                f"template_cube must be 3D (n_chan, ny, nx), got {self._template.ndim}D"
            )

    @property
    def param_names(self) -> list[str]:
        return ["dx", "dy", "dv", "flux_scale"]

    @property
    def default_params(self) -> dict[str, float]:
        return {"dx": 0.0, "dy": 0.0, "dv": 0.0, "flux_scale": 1.0}

    @property
    def bounds(self) -> dict[str, tuple[float, float]]:
        nx = self._template.shape[2]
        ny = self._template.shape[1]
        half_x = nx * self._cell_size / 2
        half_y = ny * self._cell_size / 2
        n_chan = self._template.shape[0]
        return {
            "dx": (-half_x, half_x),
            "dy": (-half_y, half_y),
            "dv": (-n_chan / 2, n_chan / 2),
            "flux_scale": (0.01, 100.0),
        }

    @property
    def cell_size(self) -> float:
        return self._cell_size

    def generate_cube(self, params: dict[str, float]) -> np.ndarray:
        dx = params.get("dx", 0.0)
        dy = params.get("dy", 0.0)
        dv = params.get("dv", 0.0)
        flux_scale = params.get("flux_scale", 1.0)

        # Convert arcsec shifts to pixel shifts
        dx_pix = dx / self._cell_size
        dy_pix = dy / self._cell_size

        # Apply 3D sub-pixel shift: (channel, y, x)
        shifted = ndshift(
            self._template,
            shift=[dv, dy_pix, dx_pix],
            order=3,
            mode="constant",
            cval=0.0,
        )

        return shifted * flux_scale


class KinMSModel(ForwardModel):
    """
    Forward model using KinMS to generate kinematic cubes.

    This is a thin wrapper around the KinMS package. It requires
    ``kinms`` to be installed (``pip install uvfit[kinms]``).

    Parameters
    ----------
    xs, ys : int
        Spatial dimensions of the output cube in pixels.
    vs : int
        Number of velocity channels.
    cell_size_arcsec : float
        Pixel size in arcseconds.
    channel_width_kms : float
        Channel width in km/s.
    sbprof : callable
        Surface brightness profile function r → SB(r).
    velprof : callable
        Velocity profile function r → v(r).
    """

    def __init__(
        self,
        xs: int,
        ys: int,
        vs: int,
        cell_size_arcsec: float,
        channel_width_kms: float,
        sbprof: Any = None,
        velprof: Any = None,
    ):
        self._xs = xs
        self._ys = ys
        self._vs = vs
        self._cell_size_arcsec = cell_size_arcsec
        self._channel_width = channel_width_kms
        self._sbprof = sbprof
        self._velprof = velprof

    @property
    def param_names(self) -> list[str]:
        return ["inc", "pa", "flux", "vsys", "gas_sigma"]

    @property
    def default_params(self) -> dict[str, float]:
        return {
            "inc": 60.0,
            "pa": 0.0,
            "flux": 1.0,
            "vsys": 0.0,
            "gas_sigma": 10.0,
        }

    @property
    def bounds(self) -> dict[str, tuple[float, float]]:
        return {
            "inc": (0.0, 90.0),
            "pa": (-180.0, 180.0),
            "flux": (0.01, 1000.0),
            "vsys": (-500.0, 500.0),
            "gas_sigma": (1.0, 200.0),
        }

    @property
    def cell_size(self) -> float:
        return self._cell_size_arcsec

    def generate_cube(self, params: dict[str, float]) -> np.ndarray:
        try:
            from kinms import KinMS
        except ImportError:
            raise ImportError(
                "KinMS is required for KinMSModel. "
                "Install with: pip install uvfit[kinms]"
            )

        inc = params.get("inc", 60.0)
        pa = params.get("pa", 0.0)
        flux = params.get("flux", 1.0)
        vsys = params.get("vsys", 0.0)
        gas_sigma = params.get("gas_sigma", 10.0)

        fov_arcsec = self._xs * self._cell_size_arcsec
        vel_range = self._vs * self._channel_width

        kin = KinMS(
            xs=fov_arcsec,
            ys=fov_arcsec,
            vs=vel_range,
            cellSize=self._cell_size_arcsec,
            dv=self._channel_width,
            inc=inc,
            posAng=pa,
            gasSigma=gas_sigma,
            intFlux=flux,
            vSys=vsys,
            sbProf=self._sbprof,
            velProf=self._velprof,
        )

        cube = kin.model_cube()

        # KinMS returns (x, y, v), transpose to (v, y, x)
        if cube.ndim == 3:
            cube = np.transpose(cube, (2, 1, 0))

        return cube.astype(np.float64)
