import h5py
import numpy as np

fname = "./data/exports_mat/sub-01_preproc_export.mat"

with h5py.File(fname, "r") as f:
    print(list(f.keys()))
    data = np.array(f["data"])
    sfreq = np.array(f["sfreq"]).squeeze()
    times = np.array(f["times"]).squeeze()

print("original shape:", data.shape)   # channels x time x trials

# convert to trials x time x channels
data = np.transpose(data, (2, 1, 0))

print("fixed shape:", data.shape)      # trials x time x channels
print("sfreq:", sfreq)
print("times shape:", times.shape)

X0 = data[0]                           # first trial, time x channels
print("first trial shape:", X0.shape)