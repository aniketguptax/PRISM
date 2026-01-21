import random
from .protocols import Process, Sample

class SimpleNonUnifilarSource(Process):
    """
    A simple non-unifilar HMM (SNS-like) over binary observations.

    Hidden states: A, B
    From A:
        emit 0 and go to A with prob p
        emit 1 and go to B with prob (1-p)
    From B:
        emit 0 and go to A with prob q
        emit 1 and go to B with prob (1-q)

    The observation '0' does not uniquely determine next hidden state history, leading to
    non-unifilar predictive structure in general.
    """
    @property
    def name(self) -> str:
        return "sns"

    def __init__(self, p: float = 0.7, q: float = 0.3):
        for v in (p, q):
            if not (0.0 < v < 1.0):
                raise ValueError("p and q must be in (0,1)")
        self.p = p
        self.q = q

    def sample(self, length: int, seed: int) -> Sample:
        rng = random.Random(seed)
        A, B = 0, 1
        s = A
        x, latent = [], []

        for _ in range(length):
            latent.append(s)
            if s == A:
                if rng.random() < self.p:
                    x.append(0); s = A
                else:
                    x.append(1); s = B
            else:
                if rng.random() < self.q:
                    x.append(0); s = A
                else:
                    x.append(1); s = B

        return Sample(x=x, latent=latent)