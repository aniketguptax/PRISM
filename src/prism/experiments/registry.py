from prism.processes import GoldenMean, EvenProcess, IIDBernoulli, MarkovOrder1, MarkovOrder2, SimpleNonUnifilarSource
from prism.processes.continuous_file import ContinuousFile
from prism.processes.linear_gaussian_ssm import LinearGaussianSSM
from prism.reconstruction import OneStepGreedyMerge
from prism.reconstruction import KalmanISSReconstructor

PROCESS_REGISTRY = {
    "iid_bernoulli": IIDBernoulli,
    "markov_order_1": MarkovOrder1,
    "markov_order_2": MarkovOrder2,
    "golden_mean": GoldenMean,
    "even_process": EvenProcess,
    "sns": SimpleNonUnifilarSource,
    "continuous_file": ContinuousFile,
    "linear_gaussian_ssm": LinearGaussianSSM,
}

RECONSTRUCTOR_REGISTRY = {
    "one_step": OneStepGreedyMerge,
    "kalman_iss": KalmanISSReconstructor,
}
