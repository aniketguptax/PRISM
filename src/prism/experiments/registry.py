from prism.processes import GoldenMean, EvenProcess, IIDBernoulli, MarkovOrder1, MarkovOrder2, SimpleNonUnifilarSource
from prism.processes.continuous_file import ContinuousFile
from prism.reconstruction import OneStepGreedyMerge
from prism.reconstruction.iss_merge import GaussianPredictiveStateModel

PROCESS_REGISTRY = {
    "iid_bernoulli": IIDBernoulli,
    "markov_order_1": MarkovOrder1,
    "markov_order_2": MarkovOrder2,
    "golden_mean": GoldenMean,
    "even_process": EvenProcess,
    "sns": SimpleNonUnifilarSource,
    "continuous_file": ContinuousFile,
}

RECONSTRUCTOR_REGISTRY = {
    "one_step": OneStepGreedyMerge,
    "kalman_iss": GaussianPredictiveStateModel,
}