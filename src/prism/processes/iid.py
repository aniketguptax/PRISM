import random
from .protocols import Process, Sample


class IIDBernoulli(Process):
    @property
    def name(self) -> str:
        return "iid_bernoulli"

    def __init__(self, p_one: float = 0.5):
        if not (0.0 < p_one < 1.0):
            raise ValueError("p_one must be between 0 and 1")
        self.p = p_one

    def sample(self, length: int, seed: int) -> Sample:
        rng = random.Random(seed)
        x = [1 if rng.random() < self.p else 0 for _ in range(length)]
        return Sample(x=x, latent=None)