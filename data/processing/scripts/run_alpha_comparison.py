"""Compare the central pre-stim alpha (8-12 Hz) baseline to the multivariate
VAR(1) predictive-fit signal as a detection-discriminative feature.

For each subject we load the exported EEG epoch tensor (trials x time x
channels), select the central channels with the same `assign_channel_region`
rule used by the matched_spatial_control sweep, take the -1500 to 0 ms
pre-stimulus window, compute log alpha (8-12 Hz) bandpower per trial as the
mean log power across central channels, and merge into the central-125-375
trial-level frame so that every trial carries both `pred_fit_contrib` and
`log_alpha_central`.

We then fit four mixed-effects models with subject random intercepts:

  M0 (alpha-only):  log_alpha ~ sdt + stimamp + (1|subject)
  M1 (alpha-as-DV): log_alpha ~ sdt + stimamp + (1|subject)  — restated
  M2 (VAR + alpha): pred_fit_contrib ~ sdt + log_alpha + stimamp + (1|subject)
  M3 (alpha-as-IV): pred_fit_contrib ~ sdt + stimamp + (1|subject) on the
       same n with non-NaN alpha — sanity that the headline survives the
       slightly-restricted sample.

The headline question is: does sdt remain significant in M2 once log_alpha
is added as a covariate? If yes, the multivariate VAR signal carries
detection-relevant information that pre-stim alpha alone does not."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from eegprep import load_exported_subject, load_subject_channel_labels
from experiments.spatial import build_channel_groups
from framework.summaries import save_table
from reporting.mechanistic_dissociation import (
    DEFAULT_BASELINE_AUGMENTED_MODEL,
    DEFAULT_HYBRID_MODEL,
    DEFAULT_RAW_MODEL,
    _safe_lme_fit,
    build_trial_level_frame,
)


DEFAULT_TRIAL_SCORES_CSV = (
    "../results_baseline/region_sliding_baseline300ms_controls_focus_q4/"
    "summary_matched_spatial_control_sweep_full/central_125_375/"
    "matched_spatial_control_trial_scores.csv"
)
DEFAULT_EXPORTS_DIR = "../exports_mat"
DEFAULT_DERIVATIVES_DIR = "../ds001785/derivatives/eegprep"
DEFAULT_OUTDIR = (
    "../results_baseline/region_sliding_baseline300ms_controls_focus_q4/"
    "summary_alpha_comparison_central"
)

PRESTIM_START_MS = -1500.0
PRESTIM_END_MS = 0.0
ALPHA_LO_HZ = 8.0
ALPHA_HI_HZ = 12.0


def compute_log_alpha_per_trial(
    data: np.ndarray,
    sfreq: float,
    times_ms: np.ndarray,
    central_idx: np.ndarray,
) -> np.ndarray:
    """Return log alpha bandpower (mean over central channels) per trial."""
    pre_mask = (times_ms >= PRESTIM_START_MS) & (times_ms < PRESTIM_END_MS)
    pre_samples = data[:, pre_mask, :][:, :, central_idx]  # trials x t x chans
    n_t = pre_samples.shape[1]
    pre_samples = pre_samples - pre_samples.mean(axis=1, keepdims=True)

    freqs = np.fft.rfftfreq(n_t, d=1.0 / sfreq)
    band_mask = (freqs >= ALPHA_LO_HZ) & (freqs <= ALPHA_HI_HZ)
    if not band_mask.any():
        raise ValueError(
            f"No FFT frequencies fall inside [{ALPHA_LO_HZ}, {ALPHA_HI_HZ}] Hz "
            f"with n_t={n_t}, sfreq={sfreq}"
        )

    spectra = np.fft.rfft(pre_samples, axis=1)
    power = (spectra.real ** 2 + spectra.imag ** 2) / n_t  # trials x freq x ch
    band_power = power[:, band_mask, :].mean(axis=1)  # trials x channels
    chan_mean = band_power.mean(axis=1)  # trials
    return np.log(chan_mean + 1e-20)


def build_alpha_frame(exports_dir: Path, derivatives_dir: Path) -> pd.DataFrame:
    rows = []
    subjects = sorted(p.name.split("_")[0] for p in exports_dir.glob("sub-*_preproc_01hz_export.mat"))
    for subject in subjects:
        export_path = exports_dir / f"{subject}_preproc_01hz_export.mat"
        try:
            data, sfreq, times_ms = load_exported_subject(export_path)
            labels = load_subject_channel_labels(subject, derivatives_dir=derivatives_dir)
        except Exception as exc:
            print(f"SKIP {subject}: {exc}")
            continue

        if len(labels) != data.shape[2]:
            print(f"SKIP {subject}: label count {len(labels)} != channels {data.shape[2]}")
            continue

        groups = build_channel_groups(labels)
        if "central" not in groups:
            print(f"SKIP {subject}: no central channels detected")
            continue
        central_idx = groups["central"]

        log_alpha = compute_log_alpha_per_trial(data, sfreq, times_ms, central_idx)
        for trial_idx, val in enumerate(log_alpha):
            rows.append(
                {"subject": subject, "trial_idx": int(trial_idx), "log_alpha_central": float(val)}
            )
        print(
            f"{subject}: n_trials={data.shape[0]}, n_central_channels={len(central_idx)}, "
            f"log_alpha mean={float(log_alpha.mean()):+.3f}, sd={float(log_alpha.std()):.3f}"
        )

    return pd.DataFrame(rows)


def fit_and_extract(frame: pd.DataFrame, formula: str, label: str) -> pd.DataFrame:
    n_obs = len(frame)
    n_groups = frame["subject"].nunique()
    model = _safe_lme_fit(frame, formula, group_col="subject")
    if model is None:
        print(f"  {label}: model failed to fit")
        return pd.DataFrame()
    rows = []
    for term in model.params.index:
        if term in {"Intercept", "Group Var"}:
            continue
        ci = model.conf_int().loc[term]
        rows.append(
            {
                "model": label,
                "formula": formula,
                "term": term,
                "beta": float(model.params[term]),
                "se": float(model.bse[term]),
                "z": float(model.tvalues[term]),
                "p": float(model.pvalues[term]),
                "ci95_low": float(ci[0]),
                "ci95_high": float(ci[1]),
                "n_obs": int(n_obs),
                "n_groups": int(n_groups),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-scores-csv", default=DEFAULT_TRIAL_SCORES_CSV)
    parser.add_argument("--exports-dir", default=DEFAULT_EXPORTS_DIR)
    parser.add_argument("--derivatives-dir", default=DEFAULT_DERIVATIVES_DIR)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--focus-start-ms", type=float, default=125.0)
    parser.add_argument("--focus-end-ms", type=float, default=375.0)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    print("Loading trial scores ...")
    trial_scores = pd.read_csv(args.trial_scores_csv)
    trial_frame = build_trial_level_frame(
        trial_scores,
        raw_model=DEFAULT_RAW_MODEL,
        hybrid_model=DEFAULT_HYBRID_MODEL,
        baseline_aug_model=DEFAULT_BASELINE_AUGMENTED_MODEL,
        focus_start_ms=args.focus_start_ms,
        focus_end_ms=args.focus_end_ms,
    )
    print(f"trial_frame: n={len(trial_frame)} trials, {trial_frame['subject'].nunique()} subjects")

    print("\nComputing log alpha per trial across all subjects ...")
    alpha_frame = build_alpha_frame(Path(args.exports_dir), Path(args.derivatives_dir))
    print(f"\nalpha_frame: n={len(alpha_frame)} (subject, trial_idx) rows")

    merged = trial_frame.merge(alpha_frame, on=["subject", "trial_idx"], how="inner")
    print(
        f"merged: n={len(merged)} trials with both pred_fit_contrib and log_alpha_central"
    )

    # Standardise log_alpha within subject so that the LME beta is interpretable
    merged["log_alpha_z"] = merged.groupby("subject")["log_alpha_central"].transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-12)
    )

    print("\n=== M0: detection ~ log_alpha (alpha as predictor of hit) ===")
    m0 = fit_and_extract(merged, "sdt_int ~ log_alpha_z + stimamp", "M0_alpha_predicts_hit")
    print(m0.to_string(index=False))

    print("\n=== M1: log_alpha ~ sdt + stimamp (alpha tracks detection?) ===")
    m1 = fit_and_extract(merged, "log_alpha_z ~ sdt_int + stimamp", "M1_alpha_on_sdt")
    print(m1.to_string(index=False))

    print("\n=== M2: pred_fit ~ sdt + log_alpha + stimamp (does VAR survive alpha control?) ===")
    m2 = fit_and_extract(
        merged,
        "pred_fit_contrib ~ sdt_int + log_alpha_z + stimamp",
        "M2_pred_fit_with_alpha_covariate",
    )
    print(m2.to_string(index=False))

    print("\n=== M3: pred_fit ~ sdt + stimamp on the same merged sample (sanity) ===")
    m3 = fit_and_extract(
        merged,
        "pred_fit_contrib ~ sdt_int + stimamp",
        "M3_pred_fit_baseline_same_n",
    )
    print(m3.to_string(index=False))

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_table(merged, out_dir / "trial_frame_with_alpha.csv")
    save_table(pd.concat([m0, m1, m2, m3], ignore_index=True), out_dir / "alpha_comparison_lme.csv")
    print(f"\nSaved {out_dir / 'trial_frame_with_alpha.csv'}")
    print(f"Saved {out_dir / 'alpha_comparison_lme.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
