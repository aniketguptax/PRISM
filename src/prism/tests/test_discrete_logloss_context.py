import math

import pytest

from prism.metrics.predictive import (
    heldout_representation_states_with_context,
    log_loss,
    log_loss_with_context,
)
from prism.reconstruction.protocols import PredictiveStateModel
from prism.representations.discrete import LastK


def _toy_model(rep_to_state: dict[tuple[int, ...], int], p_next_one: dict[int, float]) -> PredictiveStateModel:
    return PredictiveStateModel(
        rep_to_state=rep_to_state,
        p_next_one=p_next_one,
        pi={0: 1.0},
        transitions={},
        sa_counts={},
        valid=True,
        invalid_reason="",
    )


def test_first_heldout_representation_crosses_train_test_boundary() -> None:
    rep = LastK(k=3)
    x_train = [0, 1, 1]
    x_test = [0, 1, 0]

    states = heldout_representation_states_with_context(x_train, x_test, rep)
    assert states
    assert states[0] == (1, 1, 0)


def test_full_context_logloss_differs_from_suffix_only_and_matches_expected() -> None:
    rep = LastK(k=2)
    x_train = [0, 1]
    x_test = [0, 1, 0]
    model = _toy_model(
        rep_to_state={
            (1, 0): 0,
            (0, 1): 1,
        },
        p_next_one={
            0: 1.0,
            1: 0.25,
        },
    )

    ll_with_context = log_loss_with_context(x_train, x_test, rep, model)
    ll_suffix_only = log_loss(x_test, rep, model, strict=True)

    expected_with_context = -math.log(0.75) / 2.0
    expected_suffix_only = -math.log(0.75)
    assert math.isclose(ll_with_context, expected_with_context, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(ll_suffix_only, expected_suffix_only, rel_tol=0.0, abs_tol=1e-12)
    assert ll_with_context != ll_suffix_only


def test_log_loss_with_context_raises_when_no_valid_terms() -> None:
    rep = LastK(k=1)
    x_train = [0, 1, 0]
    x_test = [1]
    model = _toy_model(
        rep_to_state={(0,): 0, (1,): 0},
        p_next_one={0: 0.5},
    )

    with pytest.raises(ValueError, match="No held-out one-step terms are evaluable"):
        log_loss_with_context(x_train, x_test, rep, model)
