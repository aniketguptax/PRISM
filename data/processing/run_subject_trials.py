from pathlib import Path
import numpy as np
import pandas as pd

from eegprep import load_and_prepare_subject


def placeholder_metric(X: np.ndarray) -> dict:
    # X: time x channels
    return {
        "n_time": X.shape[0],
        "n_channels": X.shape[1],
        "mean_abs": float(np.mean(np.abs(X))),
        "var": float(np.var(X)),
        "mean_l2": float(np.mean(np.linalg.norm(X, axis=1))),
    }


def main():
    inpath = Path("./data/exports_mat/sub-01_preproc_01hz_export.mat")
    outdir = Path("./data/results")
    outdir.mkdir(parents=True, exist_ok=True)

    data, sfreq, times = load_and_prepare_subject(inpath)

    results = []
    for trial_idx in range(data.shape[0]):
        X = data[trial_idx]  # time x channels
        out = placeholder_metric(X)
        out["trial_idx"] = trial_idx
        out["sfreq"] = sfreq
        out["tmin"] = float(times[0])
        out["tmax"] = float(times[-1])
        results.append(out)

    df = pd.DataFrame(results)
    outfile = outdir / "sub-01_trial_metrics.csv"
    df.to_csv(outfile, index=False)

    print(f"Processed {len(df)} trials")
    print(df.head())
    print(f"Saved to {outfile}")


if __name__ == "__main__":
    main()