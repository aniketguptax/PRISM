from prism.processes import GoldenMean, EvenProcess
from prism.reconstruction import OneStepGreedyMerge


PROCESS_REGISTRY = {
    "golden_mean": GoldenMean,
    "even_process": EvenProcess,
}

RECONSTRUCTOR_REGISTRY = {
    "one_step": OneStepGreedyMerge,
}