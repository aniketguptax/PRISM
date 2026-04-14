from pathlib import Path
import h5py
import numpy as np


def load_exported_subject(path: str):
    path = Path(path)

    with h5py.File(path, "r") as f:
        data = np.array(f["data"])
        sfreq = float(np.array(f["sfreq"]).squeeze())
        times = np.array(f["times"]).squeeze()

    return data, sfreq, times


def zscore_per_trial(data: np.ndarray) -> np.ndarray:
    # data: trials x time x channels
    mu = data.mean(axis=1, keepdims=True)
    sd = data.std(axis=1, keepdims=True) + 1e-8
    return (data - mu) / sd


def load_and_prepare_subject(path: str):
    data, sfreq, times = load_exported_subject(path)
    data = zscore_per_trial(data)
    return data, sfreq, times