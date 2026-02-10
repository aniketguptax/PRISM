from prism.processes.iid import IIDBernoulli
from prism.processes.linear_gaussian_ssm import LinearGaussianSSM
from prism.processes.wrappers import NoisyObservation


def test_noisy_observation_accepts_binary_process() -> None:
    proc = NoisyObservation(base=IIDBernoulli(p_one=0.5), flip_p=0.2)
    sample = proc.sample(length=200, seed=0)
    assert len(sample.x) == 200
    assert set(sample.x).issubset({0, 1})


def test_noisy_observation_rejects_continuous_process() -> None:
    proc = NoisyObservation(base=LinearGaussianSSM(), flip_p=0.2)
    try:
        proc.sample(length=200, seed=0)
    except TypeError as exc:
        message = str(exc)
        assert "binary *discrete* process" in message
        assert "continuous data" in message
    else:
        raise AssertionError("NoisyObservation should reject continuous-valued observations.")
