"""Causal-primitives helpers for empirical macrostate paths."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def macro_tpm_from_labels(
    labels: np.ndarray,
    *,
    smoothing: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a macro TPM and effect distribution from one label sequence."""
    labels = np.asarray(labels, dtype=int)
    if labels.ndim != 1 or labels.size < 2:
        raise ValueError("labels must be a 1D array of length >= 2.")
    if labels.min() < 0:
        raise ValueError("labels must be non-negative integers.")

    n_states = int(labels.max()) + 1
    counts = np.zeros((n_states, n_states), dtype=float)
    for src, dst in zip(labels[:-1], labels[1:]):
        counts[src, dst] += 1.0

    if smoothing > 0.0:
        counts = counts + smoothing

    row_sums = counts.sum(axis=1, keepdims=True)
    effect_counts = counts.sum(axis=0)
    effect_total = float(effect_counts.sum())
    if effect_total > 0.0:
        stationary = effect_counts / effect_total
    else:
        stationary = np.full(n_states, 1.0 / n_states)

    tpm = np.zeros_like(counts)
    for idx in range(n_states):
        if row_sums[idx, 0] > 0.0:
            tpm[idx] = counts[idx] / row_sums[idx, 0]
        else:
            tpm[idx] = stationary
    return tpm, stationary


def _entropy_bits(probabilities: np.ndarray) -> float:
    probs = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    non_zero = probs > 0.0
    if not np.any(non_zero):
        return 0.0
    return float(-np.sum(probs[non_zero] * np.log2(probs[non_zero])))


@dataclass(frozen=True)
class CausalPrimitives:
    n_states: int
    determinism: float
    specificity: float
    cp: float


def causal_primitives(
    tpm: np.ndarray,
    *,
    intervention_distribution: np.ndarray | None = None,
) -> CausalPrimitives:
    """Compute determinism, specificity, and CP on a row-stochastic TPM."""
    tpm = np.asarray(tpm, dtype=float)
    if tpm.ndim != 2 or tpm.shape[0] != tpm.shape[1]:
        raise ValueError(f"tpm must be square, got {tpm.shape}.")

    n_states = tpm.shape[0]
    if n_states < 2:
        return CausalPrimitives(1, 1.0, 1.0, 1.0)

    if intervention_distribution is None:
        interventions = np.full(n_states, 1.0 / n_states)
    else:
        interventions = np.asarray(intervention_distribution, dtype=float)
        if interventions.shape != (n_states,):
            raise ValueError(
                f"intervention_distribution shape {interventions.shape} != ({n_states},)"
            )
        total = float(interventions.sum())
        if total <= 0.0:
            interventions = np.full(n_states, 1.0 / n_states)
        else:
            interventions = interventions / total

    log2_n = math.log2(n_states)
    row_entropies = np.array([_entropy_bits(tpm[idx]) for idx in range(n_states)])
    determinism = 1.0 - float(np.dot(interventions, row_entropies)) / log2_n
    effect_distribution = interventions @ tpm
    specificity = 1.0 - _entropy_bits(effect_distribution) / log2_n
    cp = max(0.0, determinism + specificity - 1.0)

    return CausalPrimitives(
        n_states=n_states,
        determinism=float(np.clip(determinism, 0.0, 1.0)),
        specificity=float(np.clip(specificity, 0.0, 1.0)),
        cp=float(np.clip(cp, 0.0, 1.0)),
    )


@dataclass(frozen=True)
class PathRung:
    n_states: int
    cp: float
    determinism: float
    specificity: float


def cp_path_from_label_chain(
    label_chain: Sequence[np.ndarray],
    *,
    use_observed_distribution: bool = True,
    smoothing: float = 0.0,
) -> list[PathRung]:
    """Evaluate CP at each rung of a fine-to-coarse label chain."""
    rungs: list[PathRung] = []
    for labels in label_chain:
        tpm, stationary = macro_tpm_from_labels(labels, smoothing=smoothing)
        primitives = causal_primitives(
            tpm,
            intervention_distribution=stationary if use_observed_distribution else None,
        )
        rungs.append(
            PathRung(
                n_states=primitives.n_states,
                cp=primitives.cp,
                determinism=primitives.determinism,
                specificity=primitives.specificity,
            )
        )
    return rungs


def delta_cp(rungs: Sequence[PathRung]) -> list[float]:
    """Return CP changes between successive rungs."""
    if not rungs:
        return []
    return [rungs[idx].cp - rungs[idx - 1].cp for idx in range(1, len(rungs))]


def emergent_complexity(deltas: Iterable[float], *, normalise: bool = True) -> float:
    """Return Hoel's EC from the positive part of delta-CP."""
    positive = [delta for delta in deltas if delta > 0.0]
    if len(positive) <= 1:
        return 0.0

    total = float(sum(positive))
    if total <= 0.0:
        return 0.0

    probabilities = [delta / total for delta in positive]
    raw = float(-sum(prob * math.log2(prob) for prob in probabilities))
    if not normalise:
        return raw
    return raw / math.log2(len(positive))


def total_ce(rungs: Sequence[PathRung]) -> float:
    """Return the sum of positive CP gains along the path."""
    return float(sum(delta for delta in delta_cp(rungs) if delta > 0.0))
