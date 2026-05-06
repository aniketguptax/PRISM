from prism.processes import (
    BlockModularLGSSM,
    EvenProcess,
    GoldenMean,
    HierarchicalPredictiveHMM,
    HierarchicalSwitchingGaussian,
    IIDBernoulli,
    MarkovOrder1,
    MarkovOrder2,
    MultiscaleLGSSM,
    PredictiveLowVarianceLGSSM,
    SimpleNonUnifilarSource,
)
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
    "block_modular_lgssm": BlockModularLGSSM,
    "hierarchical_predictive_hmm": HierarchicalPredictiveHMM,
    "hierarchical_switching_gaussian": HierarchicalSwitchingGaussian,
    "multiscale_lgssm": MultiscaleLGSSM,
    "predictive_low_variance_lgssm": PredictiveLowVarianceLGSSM,
}

RECONSTRUCTOR_REGISTRY = {
    "one_step": OneStepGreedyMerge,
    "kalman_iss": KalmanISSReconstructor,
}
