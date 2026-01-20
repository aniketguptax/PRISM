import random
from .protocols import Process, Sample


class MarkovOrder1(Process):
    """
    Two-state Markov chain over the observable symbol itself:
      P(X_{t+1}=1 | X_t=0) = p01
      P(X_{t+1}=1 | X_t=1) = p11
    """
    @property
    def name(self) -> str:
        return "markov_order1"

    def __init__(self, p01: float = 0.8, p11: float = 0.2):
        for p in (p01, p11):
            if not (0.0 < p < 1.0):
                raise ValueError("p01 and p11 must be between 0 and 1")
        self.p01 = p01
        self.p11 = p11

    def sample(self, length: int, seed: int) -> Sample:
        rng = random.Random(seed)
        x = []
        # initial symbol
        x0 = 1 if rng.random() < 0.5 else 0
        x.append(x0)

        for _ in range(length - 1):
            prev = x[-1]
            p1 = self.p11 if prev == 1 else self.p01
            x.append(1 if rng.random() < p1 else 0)

        return Sample(x=x, latent=None)