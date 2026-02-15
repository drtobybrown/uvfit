"""
uvfit — High-performance 3D spectral-line fitting to interferometric visibilities.
"""

__version__ = "0.1.0"

from uvfit.uvdataset import UVDataset
from uvfit.forward_model import ForwardModel, TemplateCubeModel, KinMSModel
from uvfit.nufft import NUFFTEngine
from uvfit.likelihood import VisibilityLikelihood
from uvfit.fitter import Fitter, FitResult

__all__ = [
    "UVDataset",
    "ForwardModel",
    "TemplateCubeModel",
    "KinMSModel",
    "NUFFTEngine",
    "VisibilityLikelihood",
    "Fitter",
    "FitResult",
]
