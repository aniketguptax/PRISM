"""Robustness checks for the central 125-375 ms detection LME headline:

  (1) Leave-one-subject-out: refit the LME after dropping each subject and
      report the range of betas, z-values and p-values. Headline survives if
      every leave-one-out fit remains highly significant.

  (2) Within-subject permutation null on sdt: shuffle the hit/miss label
      within subject N times, refit the LME, build the empirical null
      distribution of the sdt beta. Empirical p = fraction of null betas
      with absolute value >= observed beta.

We use specification A (`pred_fit_contrib ~ sdt_int + stimamp + (1|subject)`)
on the central 125-375 ms trial-scores CSV from the matched-spatial-control
sweep, and reuse `build_trial_level_frame` + `_safe_lme_fit` from the
mechanistic_dissociation module so the test is identical to the headline.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

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
DEFAULT_OUTDIR = (
    "../results_baseline/region_sliding_baseline300ms_controls_focus_q4/"
    "summary_mechanistic_robustness_central"
)
SDT_TERM = "sdt_int"
FORMULA = "pred_fit_contrib ~ sdt_int + stimamp"


def _fit_sdt_beta(frame: pd.DataFrame) -> tuple[float, float, float, float, float]:
    model = _safe_lme_fit(frame, FORMULA, group_col="subject")
    if model is None or SDT_TERM not in model.params.index:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    beta = float(model.params[SDT_TERM])
    se = float(model.bse[SDT_TERM])
    z = float(model.tvalues[SDT_TERM])
    p = float(model.pvalues[SDT_TERM])
    ci = model.conf_int().loc[SDT_TERM]
    ci_low = float(ci[0])
    ci_high = float(ci[1])
    return beta, se, z, p, (ci_low + ci_high) / 2  # placeholder unused


def loso(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    subjects = sorted(frame["subject"].unique())
    for sub in subjects:
        sub_frame = frame.loc[frame["subject"] != sub].copy()
        beta, se, z, p, _ = _fit_sdt_beta(sub_frame)
        rows.append(
            {
                "dropped_subject": sub,
                "n_trials": int(len(sub_frame)),
                "n_subjects": int(sub_frame["subject"].nunique()),
                "beta": beta,
                "se": se,
                "z": z,
                "p": p,
            }
        )
    return pd.DataFrame(rows)


def permutation_null(
    frame: pd.DataFrame, n_perm: int, seed: int
) -> tuple[float, float, np.ndarray]:
    rng = np.random.default_rng(seed)
    observed_beta, _, _, _, _ = _fit_sdt_beta(frame)

    null_betas = np.empty(n_perm, dtype=float)
    grouped = frame.groupby("subject", sort=False)
    for i in range(n_perm):
        permuted = frame.copy()
        # within-subject permutation of the sdt label
        new_sdt = np.empty(len(permuted), dtype=int)
        for sub, idx in grouped.groups.items():
            sdt_vals = permuted.loc[idx, "sdt_int"].to_numpy()
            permuted_sdt = rng.permutation(sdt_vals)
            new_sdt[idx.to_numpy()] = permuted_sdt
        permuted["sdt_int"] = new_sdt
        beta_i, _, _, _, _ = _fit_sdt_beta(permuted)
        null_betas[i] = beta_i
        if (i + 1) % 25 == 0 or i + 1 == n_perm:
            print(f"  perm {i + 1}/{n_perm}  null beta median={np.nanmedian(null_betas[: i + 1]):+.5f}")

    n_extreme = int(np.sum(np.abs(null_betas) >= abs(observed_beta)))
    empirical_p = (n_extreme + 1) / (n_perm + 1)
    return observed_beta, empirical_p, null_betas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-scores-csv", default=DEFAULT_TRIAL_SCORES_CSV)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--focus-start-ms", type=float, default=125.0)
    parser.add_argument("--focus-end-ms", type=float, default=375.0)
    parser.add_argument("--n-perm", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    trial_scores = pd.read_csv(args.trial_scores_csv)
    frame = build_trial_level_frame(
        trial_scores,
        raw_model=DEFAULT_RAW_MODEL,
        hybrid_model=DEFAULT_HYBRID_MODEL,
        baseline_aug_model=DEFAULT_BASELINE_AUGMENTED_MODEL,
        focus_start_ms=args.focus_start_ms,
        focus_end_ms=args.focus_end_ms,
    )
    print(
        f"Loaded trial frame: n={len(frame)} trials, "
        f"{frame['subject'].nunique()} subjects, "
        f"{int((frame['sdt_int']==1).sum())} hits"
    )

    obs_beta, obs_se, obs_z, obs_p, _ = _fit_sdt_beta(frame)
    print(
        f"\nFull-sample LME: beta={obs_beta:+.5f}  se={obs_se:.5f}  "
        f"z={obs_z:.2f}  p={obs_p:.3e}"
    )

    print("\n=== Leave-one-subject-out ===")
    loso_df = loso(frame)
    print(loso_df.to_string(index=False))
    print(
        f"\nLOSO beta range: [{loso_df['beta'].min():+.5f}, {loso_df['beta'].max():+.5f}]  "
        f"z range: [{loso_df['z'].min():.2f}, {loso_df['z'].max():.2f}]  "
        f"max p: {loso_df['p'].max():.3e}"
    )

    print(f"\n=== Within-subject permutation null (n={args.n_perm}) ===")
    obs_beta_perm, empirical_p, null_betas = permutation_null(
        frame, n_perm=int(args.n_perm), seed=int(args.seed)
    )
    print(
        f"Observed beta = {obs_beta_perm:+.5f}\n"
        f"Null beta mean={np.nanmean(null_betas):+.5f}, "
        f"std={np.nanstd(null_betas):.5f}, "
        f"|null| max={np.nanmax(np.abs(null_betas)):.5f}\n"
        f"Empirical p = {empirical_p:.4f}  ({int(np.sum(np.abs(null_betas) >= abs(obs_beta_perm)))} of {args.n_perm} permutations exceeded)"
    )

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_table(loso_df, out_dir / "loso_table.csv")
    pd.DataFrame({"perm_index": np.arange(len(null_betas)), "null_beta": null_betas}).to_csv(
        out_dir / "permutation_null_betas.csv", index=False
    )

    summary = pd.DataFrame(
        [
            {
                "test": "full_sample_lme",
                "beta": obs_beta,
                "se": obs_se,
                "z": obs_z,
                "p": obs_p,
                "n_trials": int(len(frame)),
                "n_subjects": int(frame["subject"].nunique()),
                "notes": "headline central 125-375 LME spec A",
            },
            {
                "test": "loso_min_z",
                "beta": float(loso_df["beta"].min()),
                "se": np.nan,
                "z": float(loso_df["z"].min()),
                "p": float(loso_df["p"].max()),
                "n_trials": int(loso_df["n_trials"].min()),
                "n_subjects": int(loso_df["n_subjects"].min()),
                "notes": "worst-case LOSO fit",
            },
            {
                "test": "loso_beta_range",
                "beta": float(loso_df["beta"].mean()),
                "se": float(loso_df["beta"].std()),
                "z": np.nan,
                "p": np.nan,
                "n_trials": np.nan,
                "n_subjects": np.nan,
                "notes": f"min={loso_df['beta'].min():+.5f} max={loso_df['beta'].max():+.5f}",
            },
            {
                "test": "permutation_null",
                "beta": obs_beta_perm,
                "se": np.nan,
                "z": np.nan,
                "p": empirical_p,
                "n_trials": int(len(frame)),
                "n_subjects": int(frame["subject"].nunique()),
                "notes": f"within-subject permutation, n_perm={args.n_perm}",
            },
        ]
    )
    save_table(summary, out_dir / "robustness_summary.csv")
    print(f"\nSaved {out_dir / 'loso_table.csv'}")
    print(f"Saved {out_dir / 'permutation_null_betas.csv'}")
    print(f"Saved {out_dir / 'robustness_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
