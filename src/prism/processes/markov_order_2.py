import random
from typing import Dict, Tuple

from .protocols import Process, Sample


class MarkovOrder2(Process):
    """
    Order-2 binary Markov process:
      P(X_{t+1}=1 | X_{t-1}=a, X_t=b) = p[(a,b)]
    where (a,b) in {(0,0),(0,1),(1,0),(1,1)}.
    """

    @property
    def name(self) -> str:
        return "markov_order2"

    def __init__(self, p: Dict[Tuple[int, int], float] | None = None):
        # default: deliberately different futures across contexts
        if p is None:
            p = {
                (0, 0): 0.10,
                (0, 1): 0.80,
                (1, 0): 0.70,
                (1, 1): 0.20,
            }
        for k, v in p.items():
            if k not in {(0, 0), (0, 1), (1, 0), (1, 1)}:
                raise ValueError(f"bad key {k}, must be a pair of bits")
            if not (0.0 < v < 1.0):
                raise ValueError(f"p{ k } must be in (0,1), got {v}")
        self.p = dict(p)

    def sample(self, length: int, seed: int) -> Sample:
        rng = random.Random(seed)
        # initialise with two random bits
        x0 = 1 if rng.random() < 0.5 else 0
        x1 = 1 if rng.random() < 0.5 else 0
        x = [x0, x1]

        for _ in range(length - 2):
            a, b = x[-2], x[-1]
            p1 = self.p[(a, b)]
            x.append(1 if rng.random() < p1 else 0)

        return Sample(x=x, latent=None)