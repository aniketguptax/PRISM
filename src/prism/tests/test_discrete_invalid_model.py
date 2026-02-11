import math

import pytest

from prism.metrics.predictive import log_loss
from prism.reconstruction.one_step_merge import OneStepGreedyMerge
from prism.representations.discrete import LastK


def test_oversized_k_marks_model_invalid_and_logloss_is_nan() -> None:
    x_train = [0, 1, 0, 1]
    rep = LastK(k=10)
    model = OneStepGreedyMerge(eps=0.02, strict=False).fit(x_train, rep, seed=0)
    assert not model.valid
    assert "k too large / no supported contexts" in model.invalid_reason

    ll = log_loss(x_train, rep, model)
    assert math.isnan(ll)


def test_oversized_k_strict_mode_raises() -> None:
    x_train = [0, 1, 0, 1]
    rep = LastK(k=10)
    with pytest.raises(ValueError, match="k too large / no supported contexts"):
        OneStepGreedyMerge(eps=0.02, strict=True).fit(x_train, rep, seed=0)


def test_log_loss_strict_raises_when_no_timesteps_are_evaluated() -> None:
    x = [0]
    rep = LastK(k=1)
    model = OneStepGreedyMerge(eps=0.02).fit([0, 1, 0, 1], rep, seed=0)
    with pytest.raises(ValueError, match="no evaluated timesteps"):
        log_loss(x, rep, model, strict=True)
