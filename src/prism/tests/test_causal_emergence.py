import math

import numpy as np

from prism.analysis.causal_emergence import (
    PathRung,
    causal_primitives,
    emergent_complexity,
    macro_tpm_from_labels,
    total_ce,
)


def test_macro_tpm_fills_unvisited_rows_with_effect_distribution():
    labels = np.array([0, 1, 0, 1], dtype=int)
    tpm, stationary = macro_tpm_from_labels(labels)

    assert tpm.shape == (2, 2)
    assert np.allclose(tpm.sum(axis=1), 1.0)
    assert np.allclose(stationary, np.array([1.0 / 3.0, 2.0 / 3.0]))


def test_identity_tpm_with_uniform_interventions_has_zero_specificity():
    tpm = np.eye(3, dtype=float)
    primitives = causal_primitives(tpm)

    assert primitives.n_states == 3
    assert math.isclose(primitives.determinism, 1.0)
    assert math.isclose(primitives.specificity, 0.0)
    assert math.isclose(primitives.cp, 0.0)


def test_total_ce_and_ec_use_positive_gains_only():
    rungs = [
        PathRung(n_states=8, cp=0.10, determinism=0.0, specificity=0.0),
        PathRung(n_states=4, cp=0.30, determinism=0.0, specificity=0.0),
        PathRung(n_states=2, cp=0.50, determinism=0.0, specificity=0.0),
    ]

    assert math.isclose(total_ce(rungs), 0.40)
    assert math.isclose(emergent_complexity([0.20, 0.20], normalise=True), 1.0)
