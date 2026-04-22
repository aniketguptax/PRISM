"""Loader functions for MATLAB-exported EEGPREP subject files."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat


DEFAULT_DERIVATIVES_DIR = Path("./data/ds001785/derivatives/eegprep")


def load_exported_subject(path: str | Path) -> tuple[np.ndarray, float, np.ndarray]:
    """Load one exported subject file as trials x time x channels."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Exported subject file not found: {path}")

    with h5py.File(path, "r") as f:
        data = np.array(f["data"])
        sfreq = float(np.array(f["sfreq"]).squeeze())
        times = np.array(f["times"]).squeeze()

    return data, sfreq, times


def zscore_per_trial(data: np.ndarray) -> np.ndarray:
    """Z-score each trial independently over time."""
    mu = data.mean(axis=1, keepdims=True)
    sd = data.std(axis=1, keepdims=True) + 1e-8
    return (data - mu) / sd


def load_and_prepare_subject(path: str | Path) -> tuple[np.ndarray, float, np.ndarray]:
    """Load one subject file and apply per-trial z-scoring."""
    data, sfreq, times = load_exported_subject(path)
    data = zscore_per_trial(data)
    return data, sfreq, times


def resolve_preproc_set_path(
    subject: str,
    derivatives_dir: str | Path = DEFAULT_DERIVATIVES_DIR,
) -> Path:
    derivatives_dir = Path(derivatives_dir)
    return derivatives_dir / subject / "ses-01" / "eeg" / f"{subject}_preproc_01hz.set"


def load_subject_channel_labels(
    subject: str,
    derivatives_dir: str | Path = DEFAULT_DERIVATIVES_DIR,
) -> list[str]:
    set_path = resolve_preproc_set_path(subject, derivatives_dir=derivatives_dir)
    if not set_path.exists():
        raise FileNotFoundError(f"Preprocessed EEGLAB file not found: {set_path}")

    eeg = loadmat(set_path, struct_as_record=False, squeeze_me=True)["EEG"]
    chanlocs = np.atleast_1d(eeg.chanlocs)
    labels = [str(getattr(chan, "labels", "")).strip() for chan in chanlocs]
    labels = [label for label in labels if label]

    if not labels:
        raise ValueError(f"No channel labels found in {set_path}")

    return labels
