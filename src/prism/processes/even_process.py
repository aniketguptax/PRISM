import random
from typing import List

from .protocols import Process, Sample


class EvenProcess(Process):
    @property
    def name(self) -> str:
        return "even_process"
    
    def __init__(self, p_emit_one: float = 0.7):
        if not (0.0 < p_emit_one < 1.0):
            raise ValueError("p_emit_one must be in (0, 1)")
        self.p = p_emit_one

    def sample(self, length: int, seed: int) -> Sample:
        rng = random.Random(seed)
        E, O = 0, 1
        state = E

        x: List[int] = []
        latent: List[int] = []

        for _ in range(length):
            latent.append(state)
            if state == E:
                if rng.random() < self.p:
                    x.append(1)
                    state = O
                else:
                    x.append(0)
                    state = E
            else:
                x.append(1)
                state = E

        return Sample(x=x, latent=latent)