import math
from typing import Hashable, Iterable, Iterator, NamedTuple, Optional, Sequence

from prism.reconstruction.protocols import PredictiveStateModel
from prism.representations.protocols import Representation
from prism.types import Obs

Rep = Hashable


class _EvalTerm(NamedTuple):
    rep_state: Rep
    y_next: int


def _binary_symbol(value: Obs, *, index: int) -> int:
    if isinstance(value, bool):
        out = int(value)
    elif isinstance(value, int):
        out = value
    else:
        raise TypeError(
            f"log_loss expects binary integer observations, got {type(value).__name__} at index {index}."
        )
    if out not in (0, 1):
        raise ValueError(f"log_loss expects observations in {{0,1}}, got {out} at index {index}.")
    return out


def _iter_eval_terms(
    x: Sequence[Obs],
    rep: Representation,
    *,
    start_t: int,
) -> Iterator[_EvalTerm]:
    """Yield (representation, next-symbol) terms for one-step discrete evaluation."""
    if start_t < 0:
        raise ValueError(f"start_t must be >= 0, got {start_t}.")
    if start_t > len(x):
        raise ValueError(f"start_t must be <= len(x), got {start_t} for sequence length {len(x)}.")

    for t in range(start_t, len(x) - 1):
        if t < rep.lookback:
            continue
        yield _EvalTerm(
            rep_state=rep(x, t),
            y_next=_binary_symbol(x[t + 1], index=t + 1),
        )


def _average_bernoulli_nll(
    terms: Iterable[_EvalTerm],
    model: PredictiveStateModel,
    *,
    backoff_p: float,
    on_empty: Optional[str],
) -> float:
    eps = 1e-12
    total_loss = 0.0
    n_terms = 0
    for rep_state, y_next in terms:
        s = model.rep_to_state.get(rep_state)
        p1 = model.p_next_one[s] if s is not None else backoff_p
        p = p1 if y_next == 1 else (1.0 - p1)
        total_loss += -math.log(max(p, eps))
        n_terms += 1

    if n_terms == 0:
        if on_empty is not None:
            raise ValueError(on_empty)
        return math.nan
    return total_loss / n_terms


def heldout_representation_states_with_context(
    x_context: Sequence[Obs],
    x_eval: Sequence[Obs],
    rep: Representation,
) -> list[Rep]:
    """Representational states used for held-out evaluation"""
    x_full = tuple(x_context) + tuple(x_eval)
    split = len(x_context)
    return [term.rep_state for term in _iter_eval_terms(x_full, rep, start_t=split)]


def log_loss(
    x: Sequence[Obs],
    rep: Representation,
    model: PredictiveStateModel,
    backoff_p: float = 0.5,
    *,
    strict: bool = False,
) -> float:
    """Average one-step Bernoulli negative log-likelihood.

    Returns NaN when no test terms are evaluable unless strict=True, in
    which case a ValueError is raised.
    """
    if not 0.0 <= backoff_p <= 1.0:
        raise ValueError(f"backoff_p must be in [0, 1], got {backoff_p}.")
    if not model.valid:
        reason = model.invalid_reason or "invalid predictive model"
        if strict:
            raise ValueError(f"log_loss undefined: {reason}.")
        return math.nan

    return _average_bernoulli_nll(
        _iter_eval_terms(x, rep, start_t=rep.lookback),
        model,
        backoff_p=backoff_p,
        on_empty="log_loss undefined: no evaluated timesteps." if strict else None,
    )


def log_loss_with_context(
    x_context: Sequence[Obs],
    x_eval: Sequence[Obs],
    rep: Representation,
    model: PredictiveStateModel,
    backoff_p: float = 0.5,
) -> float:
    if not 0.0 <= backoff_p <= 1.0:
        raise ValueError(f"backoff_p must be in [0, 1], got {backoff_p}.")
    if not model.valid:
        reason = model.invalid_reason or "invalid predictive model"
        raise ValueError(
            "log_loss_with_context requires a valid predictive model "
            f"(got invalid model: {reason})."
        )

    x_full = tuple(x_context) + tuple(x_eval)
    split = len(x_context)
    return _average_bernoulli_nll(
        _iter_eval_terms(x_full, rep, start_t=split),
        model,
        backoff_p=backoff_p,
        on_empty=(
            "No held-out one-step terms are evaluable. "
            "Need at least two held-out samples so x_{t+1} exists in the test segment."
        ),
    )
