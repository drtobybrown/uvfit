"""
Array backend abstraction for uvfit.

Provides a unified interface over NumPy/SciPy (CPU), JAX, PyTorch, and CuPy,
so the rest of the codebase stays backend-agnostic.

Usage:
    backend = get_backend("numpy")  # or "jax", "torch", "cupy"
    x = backend.array([1, 2, 3], dtype=backend.float64)
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import scipy.fft


@dataclass(frozen=True)
class Backend:
    """A namespace of array operations for a specific backend."""

    name: str

    # Core array creation
    array: Callable
    zeros: Callable
    ones: Callable
    arange: Callable
    linspace: Callable
    meshgrid: Callable
    stack: Callable
    concatenate: Callable

    # Math
    exp: Callable
    log: Callable
    sqrt: Callable
    abs: Callable
    sum: Callable
    real: Callable
    imag: Callable
    conj: Callable
    pi: float

    # FFT
    fft2: Callable
    ifft2: Callable
    fftshift: Callable
    ifftshift: Callable
    fftfreq: Callable

    # Data types
    float32: Any
    float64: Any
    complex64: Any
    complex128: Any

    # Utilities
    to_numpy: Callable  # convert backend array → numpy


def _numpy_backend() -> Backend:
    """Create a NumPy/SciPy backend."""
    return Backend(
        name="numpy",
        array=np.array,
        zeros=np.zeros,
        ones=np.ones,
        arange=np.arange,
        linspace=np.linspace,
        meshgrid=np.meshgrid,
        stack=np.stack,
        concatenate=np.concatenate,
        exp=np.exp,
        log=np.log,
        sqrt=np.sqrt,
        abs=np.abs,
        sum=np.sum,
        real=np.real,
        imag=np.imag,
        conj=np.conj,
        pi=np.pi,
        fft2=scipy.fft.fft2,
        ifft2=scipy.fft.ifft2,
        fftshift=scipy.fft.fftshift,
        ifftshift=scipy.fft.ifftshift,
        fftfreq=scipy.fft.fftfreq,
        float32=np.float32,
        float64=np.float64,
        complex64=np.complex64,
        complex128=np.complex128,
        to_numpy=np.asarray,
    )


def _jax_backend() -> Backend:
    """Create a JAX backend."""
    import jax.numpy as jnp
    import jax

    return Backend(
        name="jax",
        array=jnp.array,
        zeros=jnp.zeros,
        ones=jnp.ones,
        arange=jnp.arange,
        linspace=jnp.linspace,
        meshgrid=jnp.meshgrid,
        stack=jnp.stack,
        concatenate=jnp.concatenate,
        exp=jnp.exp,
        log=jnp.log,
        sqrt=jnp.sqrt,
        abs=jnp.abs,
        sum=jnp.sum,
        real=jnp.real,
        imag=jnp.imag,
        conj=jnp.conj,
        pi=jnp.pi,
        fft2=jnp.fft.fft2,
        ifft2=jnp.fft.ifft2,
        fftshift=jnp.fft.fftshift,
        ifftshift=jnp.fft.ifftshift,
        fftfreq=jnp.fft.fftfreq,
        float32=jnp.float32,
        float64=jnp.float64,
        complex64=jnp.complex64,
        complex128=jnp.complex128,
        to_numpy=lambda x: np.asarray(x),
    )


# Cache backends to avoid repeated imports
_BACKEND_CACHE: dict[str, Backend] = {}


def get_backend(name: str = "numpy") -> Backend:
    """
    Get an array backend by name.

    Parameters
    ----------
    name : str
        One of "numpy", "jax". Default: "numpy".

    Returns
    -------
    Backend
        A namespace with array operations.

    Raises
    ------
    ValueError
        If the backend name is not recognized.
    ImportError
        If the requested backend is not installed.
    """
    name = name.lower()

    if name in _BACKEND_CACHE:
        return _BACKEND_CACHE[name]

    if name == "numpy":
        backend = _numpy_backend()
    elif name == "jax":
        backend = _jax_backend()
    else:
        raise ValueError(
            f"Unknown backend '{name}'. Supported: 'numpy', 'jax'."
        )

    _BACKEND_CACHE[name] = backend
    return backend
