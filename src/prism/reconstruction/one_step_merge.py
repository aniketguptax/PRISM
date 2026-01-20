from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Dict, Hashable, List, Tuple

from .protocols import PredictiveStateModel


Rep = Hashable


@dataclass(frozen=True)
class OneStepGreedyMerge:
    eps: float = 0.02
    
    @property
    def name(self) -> str:
        return "one_step_greedy_merge"

    def fit(self, x_train: List[int], rep, seed: int = 0) -> PredictiveStateModel:
        min_t = rep.lookback
        next_counts: Dict[Rep, Counter] = defaultdict(Counter)

        # Count next-symbol frequencies for each representation r_t
        for t in range(min_t, len(x_train) - 1):
            r = rep(x_train, t)
            y = x_train[t + 1]
            next_counts[r][y] += 1

        stats: List[Tuple[Rep, float, int]] = []
        for r, cnt in next_counts.items():
            total = cnt[0] + cnt[1]
            if total == 0:
                continue
            stats.append((r, cnt[1] / total, total))

        stats.sort(key=lambda x: x[1])

        rep_to_state: Dict[Rep, int] = {}
        p_next_one: Dict[int, float] = {}

        state = 0
        bin_items: List[Tuple[Rep, float, int]] = []
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

        # Build state sequence and transition counts
        state_seq: List[int] = []
        trans_counts: Dict[Tuple[int, int], Counter] = defaultdict(Counter)

        # For transitions, we need s_t and s_{t+1}, conditioned on observed sym = x_{t+1}
        for t in range(min_t, len(x_train) - 2):
            r_t = rep(x_train, t)
            r_tp1 = rep(x_train, t + 1)
            
            s_t = rep_to_state.get(r_t)
            s_tp1 = rep_to_state.get(r_tp1)
            if s_t is None or s_tp1 is None:
                continue
            
            sym = x_train[t + 1]
            state_seq.append(s_t)
            trans_counts[(s_t, sym)][s_tp1] += 1

        # Empirical occupancy
        counts = Counter(state_seq)
        total = sum(counts.values()) or 1
        pi = {s: counts[s] / total for s in counts}
        
        # Normalise transition counts to probabilities
        transitions: Dict[Tuple[int, int], Dict[int, float]] = {}
        for key, ctr in trans_counts.items():
            denom = sum(ctr.values()) or 1
            transitions[key] = {sp: c / denom for sp, c in ctr.items()}

        return PredictiveStateModel(
            rep_to_state=rep_to_state,
            p_next_one=p_next_one,
            pi=pi,
            transitions=transitions,
        )