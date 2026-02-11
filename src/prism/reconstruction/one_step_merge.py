from collections import Counter, defaultdict
from typing import Dict, Hashable, Sequence, Tuple

from prism.representations.protocols import Representation
from prism.types import Obs

from .protocols import PredictiveStateModel, Reconstructor


Rep = Hashable


class OneStepGreedyMerge(Reconstructor[PredictiveStateModel]):
    _NO_SUPPORTED_CONTEXTS = "k too large / no supported contexts"

    def __init__(self, eps: float = 0.02, *, strict: bool = False):
        if eps < 0.0:
            raise ValueError(f"eps must be >= 0.0, got {eps}.")
        self.eps = float(eps)
        self.strict = bool(strict)

    @property
    def name(self) -> str:
        return "one_step_greedy_merge"

    def _invalid_model(self, reason: str) -> PredictiveStateModel:
        if self.strict:
            raise ValueError(reason)
        return PredictiveStateModel(
            rep_to_state={},
            p_next_one={},
            pi={},
            transitions={},
            sa_counts={},
            valid=False,
            invalid_reason=reason,
        )

    def fit(self, x_train: Sequence[Obs], rep: Representation, seed: int = 0) -> PredictiveStateModel:
        del seed

        # Ensure discrete binary ints
        x_train_int: list[int] = []
        for i, v in enumerate(x_train):
            if not isinstance(v, int):
                raise TypeError(
                    f"OneStepGreedyMerge expects discrete int observations, got {type(v).__name__} at index {i}."
                )
            if v not in (0, 1):
                raise ValueError(f"OneStepGreedyMerge expects observations in {{0,1}}, got {v} at index {i}.")
            x_train_int.append(v)

        x_train = x_train_int

        min_t = rep.lookback
        if len(x_train) <= min_t:
            return self._invalid_model(self._NO_SUPPORTED_CONTEXTS)

        next_counts: Dict[Rep, Counter] = defaultdict(Counter)

        for t in range(min_t, len(x_train) - 1):
            r = rep(x_train, t)
            y = x_train[t + 1]
            next_counts[r][y] += 1

        stats: list[Tuple[Rep, float, int]] = []
        for r, cnt in next_counts.items():
            total = cnt[0] + cnt[1]
            if total == 0:
                continue
            stats.append((r, cnt[1] / total, total))

        if not stats:
            return self._invalid_model(self._NO_SUPPORTED_CONTEXTS)

        stats.sort(key=lambda x: x[1])

        rep_to_state: Dict[Rep, int] = {}
        p_next_one: Dict[int, float] = {}

        state = 0
        bin_items: list[Tuple[Rep, float, int]] = []
        anchor = None

        def flush():
            nonlocal state, bin_items
            if not bin_items:
                return
            tot = sum(n for _, _, n in bin_items)
            p = sum(p_hat * n for _, p_hat, n in bin_items) / max(tot, 1)
            p_next_one[state] = p
            for r, _, _ in bin_items:
                rep_to_state[r] = state
            state += 1
            bin_items = []

        # Greedy binning in p-hat space
        for r, p_hat, n in stats:
            if anchor is None:
                anchor = p_hat
                bin_items = [(r, p_hat, n)]
            elif abs(p_hat - anchor) <= self.eps:
                bin_items.append((r, p_hat, n))
            else:
                flush()
                anchor = p_hat
                bin_items = [(r, p_hat, n)]
        flush()

        if not rep_to_state:
            return self._invalid_model(self._NO_SUPPORTED_CONTEXTS)

        # Build state sequence and transition counts
        state_seq: list[int] = []
        trans_counts: Dict[Tuple[int, int], Counter] = defaultdict(Counter)

        # Occupancy over all defined m_t.
        for t in range(min_t, len(x_train)):
            r_t = rep(x_train, t)
            s_t = rep_to_state.get(r_t)
            if s_t is not None:
                state_seq.append(s_t)

        if not state_seq:
            return self._invalid_model(self._NO_SUPPORTED_CONTEXTS)

        # For transitions, use all t where both m_t and m_{t+1} are defined.
        for t in range(min_t, len(x_train) - 1):
            r_t = rep(x_train, t)
            r_tp1 = rep(x_train, t + 1)

            s_t = rep_to_state.get(r_t)
            s_tp1 = rep_to_state.get(r_tp1)
            if s_t is None or s_tp1 is None:
                continue

            sym = x_train[t + 1]
            trans_counts[(s_t, sym)][s_tp1] += 1

        # Empirical occupancy
        counts = Counter(state_seq)
        total = sum(counts.values())
        if total <= 0:
            return self._invalid_model(self._NO_SUPPORTED_CONTEXTS)
        pi = {s: counts[s] / total for s in counts}

        # Normalise transition counts to probabilities and build sa_counts
        transitions: Dict[Tuple[int, int], Dict[int, float]] = {}
        sa_counts: Dict[Tuple[int, int], int] = {}
        for key, ctr in trans_counts.items():
            denom = sum(ctr.values())
            if denom <= 0:
                continue
            transitions[key] = {sp: c / denom for sp, c in ctr.items()}
            sa_counts[key] = int(denom)

        return PredictiveStateModel(
            rep_to_state=rep_to_state,
            p_next_one=p_next_one,
            pi=pi,
            transitions=transitions,
            sa_counts=sa_counts,
            valid=True,
            invalid_reason="",
        )
