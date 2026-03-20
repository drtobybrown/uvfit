"""
Fitter — orchestrates the forward-modeling fitting loop.

Supports three optimization strategies:
- "L-BFGS-B" / "Nelder-Mead" / etc. via scipy.optimize.minimize
- "emcee" — MCMC sampling via the emcee package (full posterior)
- "dynesty" — nested sampling (optional, for Bayesian evidence)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
from scipy.optimize import minimize

from uvfit.forward_model import ForwardModel
from uvfit.likelihood import VisibilityLikelihood
from uvfit.nufft import NUFFTEngine
from uvfit.uvdataset import UVDataset


@dataclass
class FitResult:
    """
    Result of a fitting run.

    Attributes
    ----------
    params : dict[str, float]
        Best-fit parameter values.
    chi2 : float
        Chi-squared at the best-fit.
    reduced_chi2 : float
        Reduced chi-squared (χ²/ν).
    n_iter : int
        Number of iterations / function evaluations.
    success : bool
        Whether the optimizer reported convergence.
    method : str
        Optimization method used.
    chains : ndarray or None
        MCMC chains, shape (n_steps, n_walkers, n_params). None for
        non-MCMC methods.
    log_prob : ndarray or None
        Log-probability for each MCMC sample. None for non-MCMC methods.
    raw_result : Any
        The raw result object from the backend optimizer.
    """

    params: dict[str, float]
    chi2: float
    reduced_chi2: float
    n_iter: int
    success: bool
    method: str
    chains: np.ndarray | None = None
    log_prob: np.ndarray | None = None
    raw_result: Any = None
    autocorr_time: np.ndarray | None = None
    converged: bool | None = None


class Fitter:
    """
    Orchestrator for visibility-space model fitting.

    Composes ForwardModel → NUFFTEngine → VisibilityLikelihood into a
    single objective function, then optimizes.

    Parameters
    ----------
    uvdata : UVDataset
        The observed visibility data.
    forward_model : ForwardModel
        The model to fit.

    Examples
    --------
    >>> fitter = Fitter(uvdata=data, forward_model=model)
    >>> result = fitter.fit(initial_params={"dx": 0.1, "dy": -0.1})
    >>> print(result.params, result.reduced_chi2)
    """

    def __init__(
        self,
        uvdata: UVDataset,
        forward_model: ForwardModel,
    ):
        self.uvdata = uvdata
        self.forward_model = forward_model
        self.engine = NUFFTEngine(cell_size=forward_model.cell_size)
        self.likelihood = VisibilityLikelihood()

    def _objective(self, param_vector: np.ndarray, param_names: list[str]) -> float:
        """
        The χ² objective function: params → cube → degrid → χ².
        """
        params = dict(zip(param_names, param_vector))

        # Forward model: params → 3D cube
        cube = self.forward_model.generate_cube(params)

        # Sub-pixel spatial shift via Fourier phase ramp (preserves high-freq info)
        phase_shift = (
            (params.get("dx", 0.0), params.get("dy", 0.0))
            if ("dx" in params or "dy" in params)
            else None
        )

        # Degrid: cube + (u, v) → model visibilities
        model_vis = self.engine.degrid(
            cube=cube,
            u=self.uvdata.u,
            v=self.uvdata.v,
            freqs=self.uvdata.freqs,
            phase_shift_arcsec=phase_shift,
        )

        # Likelihood: model_vis vs observed → χ²
        chi2 = self.likelihood.chi_squared(
            model_vis=model_vis,
            observed_vis=self.uvdata.vis_data,
            weights=self.uvdata.weights,
        )

        return chi2

    def _log_prob(
        self,
        param_vector: np.ndarray,
        param_names: list[str],
        bounds: dict[str, tuple[float, float]],
        priors: dict[str, Callable] | None,
    ) -> float:
        """
        Log-posterior for MCMC: log_prior + log_likelihood.
        """
        params = dict(zip(param_names, param_vector))

        # Flat prior within bounds
        for name, value in params.items():
            if name in bounds:
                lo, hi = bounds[name]
                if not (lo <= value <= hi):
                    return -np.inf

        # Custom priors (additive log-probability)
        log_prior = 0.0
        if priors is not None:
            for name, prior_fn in priors.items():
                if name in params:
                    log_prior += prior_fn(params[name])

        # Log-likelihood
        chi2 = self._objective(param_vector, param_names)
        log_like = -0.5 * chi2

        return log_prior + log_like

    def fit(
        self,
        initial_params: dict[str, float] | None = None,
        method: str = "L-BFGS-B",
        priors: dict[str, Callable] | None = None,
        # MCMC-specific options
        n_walkers: int = 32,
        n_steps: int = 500,
        n_burn: int = 100,
        n_processes: int = 1,
        # Tau-based convergence (emcee only)
        converge: bool = False,
        check_interval: int = 500,
        tau_factor: float = 50.0,
        tau_rtol: float = 0.01,
        max_steps: int = 10000,
        **kwargs,
    ) -> FitResult:
        """
        Run the fitting procedure.

        Parameters
        ----------
        initial_params : dict, optional
            Starting parameter values. Uses model defaults if not provided.
        method : str
            Optimization method:
            - Any scipy method: "L-BFGS-B", "Nelder-Mead", "Powell", etc.
            - "emcee": MCMC sampling (requires ``emcee`` package)
            - "dynesty": Nested sampling (requires ``dynesty`` package)
        priors : dict, optional
            Custom prior functions {param_name: callable(value) → log_prob}.
            Only used for MCMC/Bayesian methods.
        n_walkers : int
            Number of MCMC walkers (emcee only). Default: 32.
        n_steps : int
            Number of MCMC steps when *converge* is False (emcee only).
        n_burn : int
            Burn-in steps to discard when *converge* is False (emcee only).
        n_processes : int
            Parallel processes for walker evaluation (emcee only).
        converge : bool
            If True, run emcee until the integrated autocorrelation time
            (tau) stabilises, ignoring *n_steps* and *n_burn*.
        check_interval : int
            Steps between convergence checks (emcee + converge only).
        tau_factor : float
            Require total steps > tau_factor * max(tau).
        tau_rtol : float
            Require relative change in tau < tau_rtol between checks.
        max_steps : int
            Hard safety cap on total steps (emcee + converge only).
        **kwargs
            Additional keyword arguments passed to the optimizer.

        Returns
        -------
        FitResult
        """
        if initial_params is None:
            initial_params = self.forward_model.default_params.copy()

        param_names = list(initial_params.keys())
        p0 = np.array([initial_params[n] for n in param_names], dtype=np.float64)
        bounds = self.forward_model.bounds

        if method.lower() == "emcee":
            return self._fit_emcee(
                p0, param_names, bounds, priors,
                n_walkers=n_walkers, n_steps=n_steps, n_burn=n_burn,
                n_processes=n_processes,
                converge=converge,
                check_interval=check_interval,
                tau_factor=tau_factor,
                tau_rtol=tau_rtol,
                max_steps=max_steps,
                **kwargs,
            )
        elif method.lower() == "dynesty":
            return self._fit_dynesty(
                p0, param_names, bounds, priors, **kwargs
            )
        else:
            return self._fit_scipy(
                p0, param_names, bounds, method=method, **kwargs
            )

    def _fit_scipy(
        self,
        p0: np.ndarray,
        param_names: list[str],
        bounds: dict[str, tuple[float, float]],
        method: str = "L-BFGS-B",
        **kwargs,
    ) -> FitResult:
        """Optimize with scipy.optimize.minimize."""
        scipy_bounds = [bounds.get(n, (None, None)) for n in param_names]

        result = minimize(
            self._objective,
            p0,
            args=(param_names,),
            method=method,
            bounds=scipy_bounds if method in ("L-BFGS-B", "SLSQP", "trust-constr") else None,
            **kwargs,
        )

        best_params = dict(zip(param_names, result.x))
        chi2 = float(result.fun)
        n_data = 2 * self.uvdata.vis_data.size
        dof = max(n_data - len(param_names), 1)

        return FitResult(
            params=best_params,
            chi2=chi2,
            reduced_chi2=chi2 / dof,
            n_iter=int(result.nfev),
            success=bool(result.success),
            method=method,
            raw_result=result,
        )

    def _fit_emcee(
        self,
        p0: np.ndarray,
        param_names: list[str],
        bounds: dict[str, tuple[float, float]],
        priors: dict[str, Callable] | None,
        n_walkers: int = 32,
        n_steps: int = 500,
        n_burn: int = 100,
        n_processes: int = 1,
        converge: bool = False,
        check_interval: int = 500,
        tau_factor: float = 50.0,
        tau_rtol: float = 0.01,
        max_steps: int = 10000,
        **kwargs,
    ) -> FitResult:
        """
        MCMC sampling with emcee.

        When *converge* is True, the sampler runs in chunks of
        *check_interval* steps and checks the integrated autocorrelation
        time (tau) after each chunk.  Sampling stops when both criteria
        are met: ``total_steps > tau_factor * max(tau)`` **and** tau has
        stabilised to within *tau_rtol* between consecutive checks.
        Burn-in is automatically set to ``2 * max(tau)``.

        When *converge* is False, the sampler runs for a fixed *n_steps*
        and discards *n_burn* as burn-in (original behaviour).
        """
        try:
            import emcee
        except ImportError:
            raise ImportError(
                "emcee is required for MCMC fitting. "
                "Install with: pip install emcee"
            )

        n_dim = len(param_names)

        rng = kwargs.pop("rng", np.random.default_rng(42))
        pos = p0 + 1e-4 * rng.standard_normal((n_walkers, n_dim))

        pool = None
        if n_processes > 1:
            from multiprocessing import Pool
            pool = Pool(n_processes)

        tau_est: np.ndarray | None = None
        did_converge: bool | None = None

        try:
            sampler = emcee.EnsembleSampler(
                n_walkers,
                n_dim,
                self._log_prob,
                args=(param_names, bounds, priors),
                pool=pool,
                **kwargs,
            )

            if converge:
                old_tau = np.full(n_dim, np.inf)
                total_steps = 0

                while total_steps < max_steps:
                    chunk = min(check_interval, max_steps - total_steps)
                    sampler.run_mcmc(
                        pos if total_steps == 0 else None,
                        chunk, progress=True,
                    )
                    total_steps += chunk

                    try:
                        tau = sampler.get_autocorr_time(quiet=True)
                    except emcee.autocorr.AutocorrError:
                        continue

                    enough_steps = total_steps > tau_factor * np.max(tau)
                    tau_stable = np.all(
                        np.abs(tau - old_tau) / old_tau < tau_rtol
                    )

                    if enough_steps and tau_stable:
                        did_converge = True
                        tau_est = tau
                        break

                    old_tau = tau.copy()

                if did_converge is None:
                    did_converge = False
                    warnings.warn(
                        f"emcee did not converge within {max_steps} steps. "
                        "Consider increasing --max-steps.",
                        stacklevel=2,
                    )
                    try:
                        tau_est = sampler.get_autocorr_time(quiet=True)
                    except emcee.autocorr.AutocorrError:
                        tau_est = None

                if tau_est is not None:
                    n_burn = int(2 * np.max(tau_est))
                else:
                    n_burn = total_steps // 5
                n_steps = total_steps
            else:
                sampler.run_mcmc(pos, n_steps, progress=True)
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        chains = sampler.get_chain(discard=n_burn)
        log_prob_chain = sampler.get_log_prob(discard=n_burn)

        flat_chains = sampler.get_chain(discard=n_burn, flat=True)
        flat_log_prob = sampler.get_log_prob(discard=n_burn, flat=True)
        best_idx = np.argmax(flat_log_prob)
        best_params = dict(zip(param_names, flat_chains[best_idx]))

        chi2 = self._objective(flat_chains[best_idx], param_names)
        n_data = 2 * self.uvdata.vis_data.size
        dof = max(n_data - n_dim, 1)

        return FitResult(
            params=best_params,
            chi2=chi2,
            reduced_chi2=chi2 / dof,
            n_iter=n_steps * n_walkers,
            success=True,
            method="emcee",
            chains=chains,
            log_prob=log_prob_chain,
            raw_result=sampler,
            autocorr_time=tau_est,
            converged=did_converge,
        )

    def _fit_dynesty(
        self,
        p0: np.ndarray,
        param_names: list[str],
        bounds: dict[str, tuple[float, float]],
        priors: dict[str, Callable] | None,
        **kwargs,
    ) -> FitResult:
        """Nested sampling with dynesty (placeholder for future implementation)."""
        raise NotImplementedError(
            "Dynesty nested sampling is planned but not yet implemented. "
            "Use method='emcee' for Bayesian posterior estimation."
        )
