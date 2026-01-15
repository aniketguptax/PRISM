import random

def seeded_rng(seed: int) -> random.Random:
    return random.Random(seed)

def bernoulli_noise(length: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randint(0, 1) for _ in range(length)]