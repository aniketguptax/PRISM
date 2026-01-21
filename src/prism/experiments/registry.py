from prism.processes import GoldenMean, EvenProcess, IIDBernoulli, MarkovOrder1, SimpleNonUnifilarSource
from prism.reconstruction import OneStepGreedyMerge

PROCESS_REGISTRY = {
    "iid_bernoulli": IIDBernoulli,
    "markov_order1": MarkovOrder1,
    "golden_mean": GoldenMean,
    "even_process": EvenProcess,
    "sns": SimpleNonUnifilarSource,
}

RECONSTRUCTOR_REGISTRY = {
    "one_step": OneStepGreedyMerge,
}