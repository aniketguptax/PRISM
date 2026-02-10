"""Kalman innovations state-space reconstruction for continuous observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from prism.continuous.iss import (
    KalmanISSConfig,
    KalmanISSModel,
    _gaussian_nll,
    fit_kalman_iss_em,
    kalman_filter,
    one_step_predictive_y,
)
from prism.representations.continuous import ISSDim
from prism.representations.protocols import Representation
from prism.types import Obs

from .protocols import Reconstructor

Array = np.ndarray


def _to_continuous_column(x: Sequence[Obs], *, name: str) -> Array:
    values: list[float] = []
    for idx, value in enumerate(x):
        if isinstance(value, bool):
            raise TypeError(f"{name} expects real-valued observations; got bool at index {idx}.")
        if isinstance(value, int):
            raise TypeError(
                f"{name} expects continuous observations (float), got int {value} at index {idx}."
            )
        if not isinstance(value, (float, np.floating)):
            raise TypeError(
                f"{name} expects real-valued observations; got {type(value).__name__} at index {idx}."
            )
        fvalue = float(value)
        if not np.isfinite(fvalue):
            raise ValueError(f"{name} requires finite observations; got {fvalue} at index {idx}.")
        values.append(fvalue)
    return np.asarray(values, dtype=float).reshape(-1, 1)


@dataclass(frozen=True)
class GaussianPredictiveStateModel:
    """Fitted linear-Gaussian ISS model with held-out predictive utilities."""

    iss: KalmanISSModel
    latent_dim: int
    obs_dim: int
    train_avg_nll: float

    def predictive_distributions(
        self,
        observations: Sequence[Obs],
        *,
        context: Optional[Sequence[Obs]] = None,
    ) -> Tuple[Array, Array]:
        """Return one-step predictive means/covariances for `observations`."""
        y_obs = _to_continuous_column(observations, name="GaussianPredictiveStateModel.predictive_distributions")
        if y_obs.shape[0] == 0:
            return (
                np.zeros((0, self.obs_dim, 1), dtype=float),
                np.zeros((0, self.obs_dim, self.obs_dim), dtype=float),
            )
        if context is not None and len(context) > 0:
            y_ctx = _to_continuous_column(context, name="GaussianPredictiveStateModel.predictive_distributions")
            y_all = np.vstack([y_ctx, y_obs])
            mu_all, s_all, _ = one_step_predictive_y(y_all, self.iss)
            start = y_ctx.shape[0]
            return mu_all[start:], s_all[start:]
        mu_obs, s_obs, _ = one_step_predictive_y(y_obs, self.iss)
        return mu_obs, s_obs

    def average_negative_log_likelihood(
        self,
        observations: Sequence[Obs],
        *,
        context: Optional[Sequence[Obs]] = None,
    ) -> float:
        """Average one-step Gaussian NLL over `observations`."""
        y_obs = _to_continuous_column(observations, name="GaussianPredictiveStateModel.average_negative_log_likelihood")
        if y_obs.shape[0] == 0:
            return 0.0
        mu_obs, s_obs = self.predictive_distributions(observations, context=context)
        losses = [_gaussian_nll(y_obs[t], mu_obs[t], s_obs[t]) for t in range(y_obs.shape[0])]
        return float(sum(losses) / len(losses))

    def filtered_state_sequence(
        self,
        observations: Sequence[Obs],
        *,
        context: Optional[Sequence[Obs]] = None,
    ) -> Tuple[Array, Array]:
        """Filtered latent state means and covariances aligned to `observations`."""
        y_obs = _to_continuous_column(observations, name="GaussianPredictiveStateModel.filtered_state_sequence")
        if y_obs.shape[0] == 0:
            return (
                np.zeros((0, self.latent_dim, 1), dtype=float),
                np.zeros((0, self.latent_dim, self.latent_dim), dtype=float),
            )
        if context is not None and len(context) > 0:
            y_ctx = _to_continuous_column(context, name="GaussianPredictiveStateModel.filtered_state_sequence")
            y_all = np.vstack([y_ctx, y_obs])
            mu_f, p_f, _, _, _ = kalman_filter(y_all, self.iss)
            start = y_ctx.shape[0]
            return mu_f[start:], p_f[start:]
        mu_f, p_f, _, _, _ = kalman_filter(y_obs, self.iss)
        return mu_f, p_f


@dataclass
class KalmanISSReconstructor(Reconstructor[GaussianPredictiveStateModel]):
    """Fit a linear-Gaussian state-space model and expose ISS predictive states."""

    em_iters: int = 50
    em_tol: float = 1e-4
    em_ridge: float = 1e-6
    min_train_samples: int = 30

    @property
    def name(self) -> str:
        return "kalman_iss"

    @property
    def eps(self) -> float:
        # Kept for backwards-compatible CSV schema.
        return float("nan")

    def fit(
        self,
        x_train: Sequence[Obs],
        rep: Representation,
        seed: int = 0,
    ) -> GaussianPredictiveStateModel:
        if not isinstance(rep, ISSDim):
            raise TypeError(
                f"KalmanISSReconstructor expects ISSDim representation, got {type(rep).__name__}."
            )
        latent_dim = rep.d

        y_train = _to_continuous_column(x_train, name="KalmanISSReconstructor.fit")
        if y_train.shape[0] < max(self.min_train_samples, 5 * latent_dim):
            raise ValueError(
                f"Need at least {max(self.min_train_samples, 5 * latent_dim)} training samples for "
                f"stable ISS fitting with latent_dim={latent_dim}; got {y_train.shape[0]}."
            )

        cfg = KalmanISSConfig(
            latent_dim=latent_dim,
            em_iters=self.em_iters,
            tol=self.em_tol,
            ridge=self.em_ridge,
            seed=seed,
        )
        model = fit_kalman_iss_em(y_train, cfg)
        mu_train, s_train, _ = one_step_predictive_y(y_train, model)
        train_losses = [_gaussian_nll(y_train[t], mu_train[t], s_train[t]) for t in range(y_train.shape[0])]
        avg_train_nll = float(sum(train_losses) / len(train_losses))

        return GaussianPredictiveStateModel(
            iss=model,
            latent_dim=latent_dim,
            obs_dim=y_train.shape[1],
            train_avg_nll=avg_train_nll,
        )
