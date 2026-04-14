from pathlib import Path
import numpy as np
import eegprep
from eegprep import load_and_prepare_subject

print(eegprep.__file__)

path = Path("./data/exports_mat/sub-01_preproc_01hz_export.mat")

data, sfreq, times = load_and_prepare_subject(path)

print("data shape:", data.shape)   # expected: (595, 768, 65)
print("sfreq:", sfreq)
print("times shape:", times.shape)

X0 = data[0]
print("trial 0 shape:", X0.shape)  # expected: (768, 65)

print("mean over time, first 5 channels:", X0.mean(axis=0)[:5])
print("std over time, first 5 channels:", X0.std(axis=0)[:5])

print("any NaN:", np.isnan(data).any())
print("any inf:", np.isinf(data).any())
print("global mean:", data.mean())
print("global std:", data.std())