from prism.processes.iid import IIDBernoulli
from prism.processes.linear_gaussian_ssm import LinearGaussianSSM
from prism.reconstruction.kalman_iss import KalmanISSReconstructor
from prism.reconstruction.one_step_merge import OneStepGreedyMerge
from prism.representations.continuous import ISSDim
from prism.representations.discrete import LastK


def test_discrete_reconstructor_accepts_sequence_input() -> None:
    sample = IIDBernoulli(p_one=0.5).sample(length=500, seed=0)
    x_seq = tuple(sample.x)
    model = OneStepGreedyMerge(eps=0.02).fit(x_seq[:400], LastK(k=2), seed=0)
    assert model.rep_to_state


def test_continuous_reconstructor_accepts_sequence_input() -> None:
    sample = LinearGaussianSSM().sample(length=600, seed=0)
    x_seq = tuple(sample.x)
    recon = KalmanISSReconstructor(em_iters=20, em_tol=1e-4, em_ridge=1e-6)
    model = recon.fit(x_seq[:500], ISSDim(d=2), seed=0)
    assert model.latent_dim == 2
