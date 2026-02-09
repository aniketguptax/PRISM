from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

from prism.continuous.iss import (
    KalmanISSConfig,
    KalmanISSModel,
    fit_kalman_iss_em,
    one_step_predictive_y,
    _gaussian_nll,
    _sym,
)

@dataclass
class GaussianPredictiveStateModel:
    """
    Continuous analogue of PredictiveStateModel.

    - macrostates are defined by bins in (mu, Sigma) predictor space
    - emissions: y_{t+1} | macrostate m ~ N(mu_m, Sigma_m)
    - transitions conditioned on sign of innovation (A2):
        sym = 0 if innovation <= 0, sym = 1 if innovation > 0
      stored as P(m_{t+1} | m_t, sym)
    """
    # ISS parameters (fitted on training) used to compute (mu_t, S_t, innov_t) at test time
    iss: KalmanISSModel

    # macrostate parameters (predictive Gaussian for next observation)
    mu: Dict[int, np.ndarray]      # state -> (p,1)
    Sigma: Dict[int, np.ndarray]   # state -> (p,p)

    # transition probabilities keyed by (state, sym)
    transitions: Dict[Tuple[int, int], Dict[int, float]]
    sa_counts: Dict[Tuple[int, int], int]

    # empirical occupancy (training)
    pi: Dict[int, float]

    # binning anchors for assignment
    # each macrostate stores an anchor (mu_anchor, logdetSigma_anchor) for greedy binning
    anchors: List[Tuple[int, float, float]]  # (state_id, mu_scalar, logdet_scalar) for p=1
    eps_mu: float
    eps_logdet: float

    obs_dim: int

    def assign_state(self, mu_t: np.ndarray, S_t: np.ndarray) -> Optional[int]:
        """
        Assign a time point to the closest (first matching) greedy bin.
        For robustness and interpretability we restrict to p=1 for now.
        """
        if self.obs_dim != 1:
            raise ValueError("assign_state currently supports obs_dim=1 only (extend if needed).")

        mu_scalar = float(mu_t.reshape(-1)[0])
        logdet = float(np.log(max(float(S_t.reshape(1, 1)[0, 0]), 1e-12)))

        # Greedy binning rule: first anchor within thresholds
        for sid, a_mu, a_ld in self.anchors:
            if abs(mu_scalar - a_mu) <= self.eps_mu and abs(logdet - a_ld) <= self.eps_logdet:
                return sid
        return None

def _innovation_sign(innov: np.ndarray) -> int:
    # A2 symbol: sign of innovation (scalar)
    return 1 if float(innov.reshape(-1)[0]) > 0.0 else 0

@dataclass
class KalmanISSGreedyMerge:
    """
    Fit Kalman ISS (EM) and then reconstruct macrostates by greedy binning in
    predictive Gaussian parameter space.

    obs assumed 1D for now: p=1.
    """
    latent_dim: int = 4
    em_iters: int = 50
    em_tol: float = 1e-4
    em_ridge: float = 1e-6

    eps_mu: float = 0.10        # threshold in predictive mean space
    eps_logdet: float = 0.25    # threshold in log-variance space

    @property
    def name(self) -> str:
        return "kalman_iss_greedy_merge"

    @property
    def eps(self) -> float:
        # keep compatibility with CSV schema; not used directly
        return float("nan")

    def fit(self, x_train: List[float], rep, seed: int = 0) -> GaussianPredictiveStateModel:
        # x_train: real-valued observations
        y = np.asarray(x_train, dtype=float).reshape(-1, 1)
        if y.shape[0] < 20:
            raise ValueError("Need at least 20 samples for ISS + reconstruction to be meaningful.")

        # 1) Fit Kalman ISS via EM
        iss_cfg = KalmanISSConfig(
            latent_dim=self.latent_dim,
            em_iters=self.em_iters,
            tol=self.em_tol,
            ridge=self.em_ridge,
            seed=seed,
        )
        iss = fit_kalman_iss_em(y, iss_cfg)

        # 2) Compute one-step predictive (mu_t, S_t) and innovations on training
        mu_y, S_y, innov = one_step_predictive_y(y, iss)  # for each t, predicts y_t|past
        # For PRISM we want predictor of y_{t+1}|past at time t.
        # Shift: use mu_y[t+1], S_y[t+1] as predictor from time t.
        mu_pred = mu_y[1:]     # (T-1, 1,1)
        S_pred = S_y[1:]       # (T-1, 1,1)
        innov_next = innov[1:] # innovation for y_{t} at index t (aligned)

        Tm1 = mu_pred.shape[0]

        # 3) Greedy binning in (mu, logvar) predictor space
        stats: List[Tuple[int, float, float]] = []
        for t in range(Tm1):
            mu_scalar = float(mu_pred[t].reshape(-1)[0])
            var = float(S_pred[t].reshape(1, 1)[0, 0])
            var = max(var, 1e-12)
            ld = float(np.log(var))
            stats.append((t, mu_scalar, ld))

        stats.sort(key=lambda a: a[1])  # sort by mu

        time_to_state: Dict[int, int] = {}
        anchors: List[Tuple[int, float, float]] = []

        state_id = 0
        bin_times: List[int] = []
        a_mu: Optional[float] = None
        a_ld: Optional[float] = None

        def flush_bin():
            nonlocal state_id, bin_times, a_mu, a_ld
            if not bin_times:
                return

            # Aggregate macrostate predictor params by averaging mu and var across assigned times
            mus = np.array([float(mu_pred[t].reshape(-1)[0]) for t in bin_times], dtype=float)
            vars_ = np.array([float(S_pred[t].reshape(1, 1)[0, 0]) for t in bin_times], dtype=float)
            var_m = float(np.mean(np.clip(vars_, 1e-12, np.inf)))
            mu_m = float(np.mean(mus))

            for t in bin_times:
                time_to_state[t] = state_id

            if a_mu is not None and a_ld is not None:
                anchors.append((state_id, float(a_mu), float(a_ld)))
            else:
                raise RuntimeError("a_mu or a_ld is None when trying to append to anchors.")
            state_id += 1
            bin_times = []
            a_mu = None
            a_ld = None

        for t, mu_s, ld_s in stats:
            if a_mu is None:
                a_mu, a_ld = mu_s, ld_s
                bin_times = [t]
            else:
                if a_mu is not None and a_ld is not None and abs(mu_s - a_mu) <= self.eps_mu and abs(ld_s - a_ld) <= self.eps_logdet:
                    bin_times.append(t)
                else:
                    flush_bin()
                    a_mu, a_ld = mu_s, ld_s
                    bin_times = [t]
        flush_bin()

        n_states = state_id
        if n_states < 1:
            raise RuntimeError("Reconstruction produced zero states unexpectedly.")

        # 4) Macrostate Gaussian parameters (predicting y_{t+1})
        mu_state: Dict[int, np.ndarray] = {}
        Sigma_state: Dict[int, np.ndarray] = {}
        for sid in range(n_states):
            ts = [t for t, s in time_to_state.items() if s == sid]
            mus = np.array([float(mu_pred[t].reshape(-1)[0]) for t in ts], dtype=float)
            vars_ = np.array([float(S_pred[t].reshape(1, 1)[0, 0]) for t in ts], dtype=float)
            mu_state[sid] = np.array([[float(np.mean(mus))]])
            Sigma_state[sid] = np.array([[float(np.mean(np.clip(vars_, 1e-12, np.inf)))]])

        # 5) Transitions conditioned on sign of innovation (A2)
        # State at time t corresponds to predictor for y_{t+1}, i.e. index t in [0..T-2]
        # Next state at time t+1 corresponds to predictor for y_{t+2}, i.e. index t+1
        trans_counts: Dict[Tuple[int, int], Dict[int, int]] = {}
        sa_counts: Dict[Tuple[int, int], int] = {}

        # occupancy over states (training)
        occ = np.zeros(n_states, dtype=int)

        for t in range(Tm1 - 1):
            s_t = time_to_state.get(t)
            s_tp1 = time_to_state.get(t + 1)
            if s_t is None or s_tp1 is None:
                continue

            occ[s_t] += 1

            sym = _innovation_sign(innov_next[t])  # sign of innovation for y_{t+1}
            key = (s_t, sym)
            trans_counts.setdefault(key, {})
            trans_counts[key][s_tp1] = trans_counts[key].get(s_tp1, 0) + 1
            sa_counts[key] = sa_counts.get(key, 0) + 1

        total_occ = int(np.sum(occ)) or 1
        pi = {s: float(occ[s]) / total_occ for s in range(n_states) if occ[s] > 0}

        transitions: Dict[Tuple[int, int], Dict[int, float]] = {}
        for key, ctr in trans_counts.items():
            denom = sum(ctr.values()) or 1
            transitions[key] = {sp: c / denom for sp, c in ctr.items()}

        return GaussianPredictiveStateModel(
            iss=iss,
            mu=mu_state,
            Sigma=Sigma_state,
            transitions=transitions,
            sa_counts=sa_counts,
            pi=pi,
            anchors=anchors,
            eps_mu=self.eps_mu,
            eps_logdet=self.eps_logdet,
            obs_dim=1,
        )