from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from prism.continuous.iss import KalmanISSModel, solve_steady_state_kalman

Array = np.ndarray


@dataclass(frozen=True)
class InnovationsForm:
    A: Array
    C: Array
    K: Array
    V: Array


@dataclass(frozen=True)
class PsiOptimisationResult:
    psi: float
    L: Array
    optimiser: str
    macro_dim: int
    restarts: int
    iterations: int


def _sym(matrix: Array) -> Array:
    return 0.5 * (matrix + matrix.T)


def _ensure_spd(matrix: Array, *, ridge: float = 1e-9, max_tries: int = 8) -> Array:
    out = _sym(matrix)
    eye = np.eye(out.shape[0], dtype=float)
    jitter = max(float(ridge), 0.0)
    for _ in range(max_tries):
        try:
            np.linalg.cholesky(out + jitter * eye)
            return out + jitter * eye
        except np.linalg.LinAlgError:
            jitter = 10.0 * jitter if jitter > 0.0 else 1e-12
    raise np.linalg.LinAlgError("Failed to make covariance SPD.")


def _normalise_rows(matrix: Array) -> Array:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _stable_logdet_spd(matrix: Array, *, ridge: float) -> float:
    spd = _ensure_spd(matrix, ridge=ridge)
    sign, logdet = np.linalg.slogdet(spd)
    if sign <= 0.0:
        raise ValueError("Expected SPD matrix with positive determinant sign.")
    return float(logdet)


def _mi_from_cov(cov: Array, cond_cov: Array, *, ridge: float) -> float:
    mi = 0.5 * (_stable_logdet_spd(cov, ridge=ridge) - _stable_logdet_spd(cond_cov, ridge=ridge))
    if mi < 0.0 and mi > -1e-6:
        return 0.0
    return max(0.0, float(mi))


def _solve_discrete_lyapunov(
    A: Array,
    Q: Array,
    *,
    tol: float = 1e-9,
    max_iter: int = 10_000,
    ridge: float = 1e-9,
) -> Array:
    """Solve O = A O A^T + Q via fixed-point iteration"""
    O = np.zeros_like(Q)
    for _ in range(max_iter):
        O_new = _sym(A @ O @ A.T + Q)
        rel = np.linalg.norm(O_new - O) / max(np.linalg.norm(O), 1e-12)
        O = O_new
        if rel < tol:
            return _ensure_spd(O, ridge=ridge)
    return _ensure_spd(O, ridge=ridge)


def _solve_filter_dare(
    A: Array,
    C: Array,
    Q: Array,
    R: Array,
    S: Array,
    *,
    tol: float = 1e-9,
    max_iter: int = 10_000,
    ridge: float = 1e-9,
) -> Array:
    """
    Steady-state covariance for the estimator Riccati equation with cross-covariance S
    """
    P = _ensure_spd(Q, ridge=ridge)
    for _ in range(max_iter):
        innovation_cov = _ensure_spd(C @ P @ C.T + R, ridge=ridge)
        cross = A @ P @ C.T + S
        P_new = _sym(A @ P @ A.T + Q - cross @ np.linalg.solve(innovation_cov, cross.T))
        rel = np.linalg.norm(P_new - P) / max(np.linalg.norm(P), 1e-12)
        P = P_new
        if rel < tol:
            return _ensure_spd(P, ridge=ridge)
    return _ensure_spd(P, ridge=ridge)


def innovations_from_ssm(
    model: KalmanISSModel,
    *,
    tol: float = 1e-9,
    max_iter: int = 10_000,
    ridge: float = 1e-9,
    strict: bool = True,
) -> InnovationsForm:
    """Convert standard SSM (A,C,Q,R) into innovation form (A,C,K,V)"""
    A = np.asarray(model.A, dtype=float)
    C = np.asarray(model.C, dtype=float)
    ss = solve_steady_state_kalman(
        model,
        tol=tol,
        max_iter=max_iter,
        ridge=ridge,
        strict=strict,
    )
    V = _ensure_spd(ss.innovation_cov, ridge=ridge)
    K = np.asarray(ss.innovations_gain, dtype=float)
    return InnovationsForm(A=A, C=C, K=K, V=V)


def compute_ss_psi(
    innovations: InnovationsForm,
    L: Array,
    *,
    tol: float = 1e-8,
    max_iter: int = 4_000,
    ridge: float = 1e-8,
) -> float:
    """
    Compute Psi for a given innovation-form model and coarse-graining L
    """
    A = np.asarray(innovations.A, dtype=float)
    C = np.asarray(innovations.C, dtype=float)
    K = np.asarray(innovations.K, dtype=float)
    V = _ensure_spd(np.asarray(innovations.V, dtype=float), ridge=ridge)
    L = np.asarray(L, dtype=float)

    micro_dim = C.shape[0]
    if L.ndim != 2 or L.shape[1] != micro_dim:
        raise ValueError(
            f"L must have shape (macro_dim, {micro_dim}), got {L.shape}."
        )

    macro_dim = L.shape[0]
    C_full = np.vstack((C, L @ C))
    Q_full = K @ V @ K.T
    R_full = np.block([[V, V @ L.T], [L @ V, L @ V @ L.T]])
    S_full = np.hstack((K @ V, K @ V @ L.T))

    macro_idx = list(range(micro_dim, micro_dim + macro_dim))
    partitions = [[i] for i in range(micro_dim)] + [macro_idx]

    conditioned_macro_covs: list[Array] = []
    for idx in partitions:
        C_sub = C_full[idx, :]
        R_sub = R_full[np.ix_(idx, idx)]
        S_sub = S_full[:, idx]
        P_sub = _solve_filter_dare(
            A,
            C_sub,
            Q_full,
            R_sub,
            S_sub,
            tol=tol,
            max_iter=max_iter,
            ridge=ridge,
        )
        residual_cov = C_full @ P_sub @ C_full.T + R_full
        conditioned_macro_covs.append(_ensure_spd(residual_cov[np.ix_(macro_idx, macro_idx)], ridge=ridge))

    state_cov = _solve_discrete_lyapunov(A, Q_full, tol=tol, max_iter=max_iter, ridge=ridge)
    full_cov = C_full @ state_cov @ C_full.T + R_full
    cov_macro = _ensure_spd(full_cov[np.ix_(macro_idx, macro_idx)], ridge=ridge)

    psi_macro = _mi_from_cov(cov_macro, conditioned_macro_covs[-1], ridge=ridge)
    psi_micro = sum(_mi_from_cov(cov_macro, cov_i, ridge=ridge) for cov_i in conditioned_macro_covs[:-1])
    return float(psi_macro - psi_micro)


def _optimise_psi_random(
    innovations: InnovationsForm,
    *,
    macro_dim: int,
    seed: int,
    restarts: int,
    iterations: int,
    step_scale: float,
    tol: float,
    max_iter: int,
    ridge: float,
) -> PsiOptimisationResult:
    rng = np.random.default_rng(seed)
    micro_dim = innovations.C.shape[0]
    if macro_dim < 1 or macro_dim > micro_dim:
        raise ValueError(f"macro_dim must be in [1, {micro_dim}], got {macro_dim}.")

    best_psi = -np.inf
    best_L = np.zeros((macro_dim, micro_dim), dtype=float)
    for _ in range(restarts):
        L = _normalise_rows(rng.normal(size=(macro_dim, micro_dim)))
        psi_val = compute_ss_psi(innovations, L, tol=tol, max_iter=max_iter, ridge=ridge)
        for step in range(iterations):
            local_scale = step_scale / np.sqrt(step + 1.0)
            candidate = _normalise_rows(L + rng.normal(scale=local_scale, size=L.shape))
            candidate_psi = compute_ss_psi(innovations, candidate, tol=tol, max_iter=max_iter, ridge=ridge)
            if candidate_psi > psi_val:
                L = candidate
                psi_val = candidate_psi
        if psi_val > best_psi:
            best_psi = psi_val
            best_L = L.copy()

    return PsiOptimisationResult(
        psi=float(best_psi),
        L=best_L,
        optimiser="random",
        macro_dim=macro_dim,
        restarts=restarts,
        iterations=iterations,
    )


def _compute_ss_psi_torch(
    A_t,
    C_t,
    K_t,
    V_t,
    L_t,
    *,
    tol: float,
    max_iter: int,
    ridge: float,
):
    import torch

    def ensure_spd_torch(matrix):
        sym = 0.5 * (matrix + matrix.T)
        evals, evecs = torch.linalg.eigh(sym)
        evals = torch.clamp(evals, min=ridge)
        return evecs @ torch.diag(evals) @ evecs.T

    def solve_filter_dare_torch(A, C, Q, R, S):
        P = ensure_spd_torch(Q)
        for _ in range(max_iter):
            innovation_cov = ensure_spd_torch(C @ P @ C.T + R)
            cross = A @ P @ C.T + S
            P_new = 0.5 * (A @ P @ A.T + Q - cross @ torch.linalg.solve(innovation_cov, cross.T) + (A @ P @ A.T + Q - cross @ torch.linalg.solve(innovation_cov, cross.T)).T)
            rel = torch.linalg.norm(P_new - P) / torch.clamp(torch.linalg.norm(P), min=1e-12)
            P = P_new
            if float(rel.detach().cpu()) < tol:
                break
        return ensure_spd_torch(P)

    def solve_lyap_torch(A, Q):
        O = torch.zeros_like(Q)
        for _ in range(max_iter):
            O_new = 0.5 * (A @ O @ A.T + Q + (A @ O @ A.T + Q).T)
            rel = torch.linalg.norm(O_new - O) / torch.clamp(torch.linalg.norm(O), min=1e-12)
            O = O_new
            if float(rel.detach().cpu()) < tol:
                break
        return ensure_spd_torch(O)

    def mi_torch(cov, cond):
        cov_spd = ensure_spd_torch(cov)
        cond_spd = ensure_spd_torch(cond)
        sign1, ld1 = torch.linalg.slogdet(cov_spd)
        sign2, ld2 = torch.linalg.slogdet(cond_spd)
        if float(sign1.detach().cpu()) <= 0.0 or float(sign2.detach().cpu()) <= 0.0:
            raise ValueError("Expected SPD covariance in torch Psi computation.")
        return torch.clamp(0.5 * (ld1 - ld2), min=0.0)

    micro_dim = C_t.shape[0]
    macro_dim = L_t.shape[0]
    C_full = torch.cat((C_t, L_t @ C_t), dim=0)
    Q_full = K_t @ V_t @ K_t.T
    upper = torch.cat((V_t, V_t @ L_t.T), dim=1)
    lower = torch.cat((L_t @ V_t, L_t @ V_t @ L_t.T), dim=1)
    R_full = torch.cat((upper, lower), dim=0)
    S_full = torch.cat((K_t @ V_t, K_t @ V_t @ L_t.T), dim=1)

    macro_idx = list(range(micro_dim, micro_dim + macro_dim))
    partitions = [[i] for i in range(micro_dim)] + [macro_idx]
    conditioned_covs = []
    for idx in partitions:
        idx_t = torch.tensor(idx, dtype=torch.long, device=C_t.device)
        C_sub = C_full.index_select(0, idx_t)
        R_sub = R_full.index_select(0, idx_t).index_select(1, idx_t)
        S_sub = S_full.index_select(1, idx_t)
        P_sub = solve_filter_dare_torch(A_t, C_sub, Q_full, R_sub, S_sub)
        residual = C_full @ P_sub @ C_full.T + R_full
        macro_idx_t = torch.tensor(macro_idx, dtype=torch.long, device=C_t.device)
        conditioned_covs.append(residual.index_select(0, macro_idx_t).index_select(1, macro_idx_t))

    state_cov = solve_lyap_torch(A_t, Q_full)
    full_cov = C_full @ state_cov @ C_full.T + R_full
    macro_idx_t = torch.tensor(macro_idx, dtype=torch.long, device=C_t.device)
    cov_macro = full_cov.index_select(0, macro_idx_t).index_select(1, macro_idx_t)
    psi_macro = mi_torch(cov_macro, conditioned_covs[-1])
    psi_micro = sum(mi_torch(cov_macro, cov_i) for cov_i in conditioned_covs[:-1])
    return psi_macro - psi_micro


def _optimise_psi_torch_adam(
    innovations: InnovationsForm,
    *,
    macro_dim: int,
    seed: int,
    restarts: int,
    iterations: int,
    lr: float,
    tol: float,
    max_iter: int,
    ridge: float,
) -> PsiOptimisationResult:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "psi_optimiser='torch_adam' requires torch. Install torch or use psi_optimiser='random'."
        ) from exc

    torch.manual_seed(seed)
    A_t = torch.tensor(innovations.A, dtype=torch.float64)
    C_t = torch.tensor(innovations.C, dtype=torch.float64)
    K_t = torch.tensor(innovations.K, dtype=torch.float64)
    V_t = torch.tensor(innovations.V, dtype=torch.float64)
    micro_dim = C_t.shape[0]
    if macro_dim < 1 or macro_dim > micro_dim:
        raise ValueError(f"macro_dim must be in [1, {micro_dim}], got {macro_dim}.")

    best_psi = -np.inf
    best_L = np.zeros((macro_dim, micro_dim), dtype=float)
    for _ in range(restarts):
        L = torch.randn((macro_dim, micro_dim), dtype=torch.float64, requires_grad=True)
        optimiser = torch.optim.Adam([L], lr=lr)
        prev_psi = None
        for _ in range(iterations):
            optimiser.zero_grad()
            psi_t = _compute_ss_psi_torch(
                A_t,
                C_t,
                K_t,
                V_t,
                L,
                tol=tol,
                max_iter=max_iter,
                ridge=ridge,
            )
            (-psi_t).backward()
            optimiser.step()
            with torch.no_grad():
                norms = torch.linalg.norm(L, dim=1, keepdim=True).clamp_min(1e-12)
                L.div_(norms)
            psi_scalar = float(psi_t.detach().cpu())
            if prev_psi is not None and abs(psi_scalar - prev_psi) <= tol * (1.0 + abs(prev_psi)):
                break
            prev_psi = psi_scalar

        final_psi = float(
            _compute_ss_psi_torch(
                A_t,
                C_t,
                K_t,
                V_t,
                L,
                tol=tol,
                max_iter=max_iter,
                ridge=ridge,
            )
            .detach()
            .cpu()
        )
        if final_psi > best_psi:
            best_psi = final_psi
            best_L = L.detach().cpu().numpy().copy()

    return PsiOptimisationResult(
        psi=float(best_psi),
        L=best_L,
        optimiser="torch_adam",
        macro_dim=macro_dim,
        restarts=restarts,
        iterations=iterations,
    )


def optimise_ss_psi(
    innovations: InnovationsForm,
    *,
    macro_dim: int = 1,
    seed: int = 0,
    optimiser: str = "random",
    restarts: int = 12,
    iterations: int = 120,
    lr: float = 0.03,
    step_scale: float = 0.2,
    tol: float = 1e-8,
    max_iter: int = 4_000,
    ridge: float = 1e-8,
) -> PsiOptimisationResult:
    if optimiser == "random":
        return _optimise_psi_random(
            innovations,
            macro_dim=macro_dim,
            seed=seed,
            restarts=restarts,
            iterations=iterations,
            step_scale=step_scale,
            tol=tol,
            max_iter=max_iter,
            ridge=ridge,
        )
    if optimiser == "torch_adam":
        return _optimise_psi_torch_adam(
            innovations,
            macro_dim=macro_dim,
            seed=seed,
            restarts=restarts,
            iterations=iterations,
            lr=lr,
            tol=tol,
            max_iter=max_iter,
            ridge=ridge,
        )
    raise ValueError(f"Unknown Psi optimiser {optimiser!r}. Expected 'random' or 'torch_adam'.")
