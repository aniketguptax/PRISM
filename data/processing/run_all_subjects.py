from pathlib import Path
import numpy as np
import pandas as pd

from eegprep import load_and_prepare_subject


def placeholder_metric(X: np.ndarray) -> dict:
    return {
        "n_time": X.shape[0],
        "n_channels": X.shape[1],
        "mean_abs": float(np.mean(np.abs(X))),
        "var": float(np.var(X)),
        "mean_l2": float(np.mean(np.linalg.norm(X, axis=1))),
    }


def extract_subject_id(path: Path) -> str:
    name = path.name
    # e.g. sub-01_preproc_01hz_export.mat -> sub-01
    return name.split("_")[0]


def run_subject(inpath: Path, outdir: Path) -> Path:
    data, sfreq, times = load_and_prepare_subject(inpath)
    subject = extract_subject_id(inpath)

    results = []
    for trial_idx in range(data.shape[0]):
        X = data[trial_idx]  # time x channels
        out = placeholder_metric(X)
        out["subject"] = subject
        out["trial_idx"] = trial_idx
        out["sfreq"] = sfreq
        out["tmin_ms"] = float(times[0])
        out["tmax_ms"] = float(times[-1])
        results.append(out)

    df = pd.DataFrame(results)
    outfile = outdir / f"{subject}_trial_metrics.csv"
    df.to_csv(outfile, index=False)
    return outfile


def main():
    export_dir = Path("./data/exports_mat")
    outdir = Path("./data/results")
    outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(export_dir.glob("*_export.mat"))
    print(f"Found {len(files)} exported subjects")

    written = []
    for fpath in files:
        try:
            outfile = run_subject(fpath, outdir)
            written.append(outfile)
            print(f"Saved {outfile}")
        except Exception as e:
            print(f"FAILED on {fpath.name}: {e}")

    if written:
        dfs = [pd.read_csv(p) for p in written]
        combined = pd.concat(dfs, ignore_index=True)
        combined_out = outdir / "all_subjects_trial_metrics.csv"
        combined.to_csv(combined_out, index=False)
        print(f"Saved combined results to {combined_out}")
        print(combined.groupby('subject').size())


if __name__ == "__main__":
    main()