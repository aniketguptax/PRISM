from pathlib import Path

from prism.processes.continuous_file import ContinuousFile, load_timeseries


def test_load_timeseries_multivariate_default(tmp_path: Path) -> None:
    csv_path = tmp_path / "series.csv"
    csv_path.write_text("1.0,2.0,3.0\n4.0,5.0,6.0\n7.0,8.0,9.0\n", encoding="utf-8")

    matrix = load_timeseries(csv_path)
    assert matrix.shape == (3, 3)

    sample = ContinuousFile(path=csv_path).sample(length=3, seed=0)
    assert len(sample.x) == 3
    first = sample.x[0]
    assert isinstance(first, tuple)
    assert len(first) == 3


def test_load_timeseries_column_subset(tmp_path: Path) -> None:
    csv_path = tmp_path / "series.csv"
    csv_path.write_text("1,2,3\n4,5,6\n", encoding="utf-8")

    matrix = load_timeseries(csv_path, columns=(0, 2))
    assert matrix.shape == (2, 2)

    sample = ContinuousFile(path=csv_path, columns=(0, 2)).sample(length=2, seed=0)
    assert sample.x == [(1.0, 3.0), (4.0, 6.0)]
