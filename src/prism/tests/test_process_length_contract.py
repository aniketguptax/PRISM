from pathlib import Path

import pytest

from prism.processes.continuous_file import ContinuousFile
from prism.processes.even_process import EvenProcess
from prism.processes.golden_mean import GoldenMean
from prism.processes.iid import IIDBernoulli
from prism.processes.linear_gaussian_ssm import LinearGaussianSSM
from prism.processes.markov_order_1 import MarkovOrder1
from prism.processes.markov_order_2 import MarkovOrder2
from prism.processes.sns import SimpleNonUnifilarSource
from prism.processes.wrappers import NoisyObservation, Subsample


@pytest.mark.parametrize(
    "process_factory",
    [
        lambda: IIDBernoulli(p_one=0.5),
        lambda: MarkovOrder1(p01=0.8, p11=0.2),
        lambda: MarkovOrder2(),
        lambda: GoldenMean(p_emit_one=0.7),
        lambda: EvenProcess(p_emit_one=0.7),
        lambda: SimpleNonUnifilarSource(p=0.7, q=0.3),
        lambda: LinearGaussianSSM(latent_dim=2, obs_dim=3),
        lambda: NoisyObservation(base=IIDBernoulli(p_one=0.5), flip_p=0.2),
        lambda: Subsample(base=IIDBernoulli(p_one=0.5), step=3),
    ],
)
@pytest.mark.parametrize("length", [1, 2, 17])
def test_processes_respect_sample_length_contract(process_factory, length: int) -> None:
    process = process_factory()
    sample = process.sample(length=length, seed=7)
    assert len(sample.x) == length
    if sample.latent is not None:
        assert len(sample.latent) == length


def test_continuous_file_respects_length_contract(tmp_path: Path) -> None:
    csv_path = tmp_path / "series.csv"
    csv_path.write_text("0.0,1.0\n1.0,2.0\n2.0,3.0\n3.0,4.0\n", encoding="utf-8")
    sample = ContinuousFile(path=csv_path).sample(length=3, seed=0)
    assert len(sample.x) == 3


def test_markov_order_2_length_1_and_2_are_exact() -> None:
    proc = MarkovOrder2()
    assert len(proc.sample(length=1, seed=123).x) == 1
    assert len(proc.sample(length=2, seed=123).x) == 2
