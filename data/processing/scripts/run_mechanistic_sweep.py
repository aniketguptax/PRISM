"""Run the trial-level mechanistic sweep across regions and time windows."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from reporting.mechanistic_dissociation import (
    DEFAULT_BASELINE_AUGMENTED_MODEL,
    DEFAULT_HYBRID_MODEL,
    DEFAULT_RAW_MODEL,
    build_trial_level_frame,
    fit_confidence_mixed_effects,
    fit_detection_mixed_effects,
    per_subject_spearman_dissociation,
    summarise_spearman_dissociation,
)
from framework.summaries import save_table


SWEEP_DIR = Path(
    "../results_baseline/region_sliding_baseline300ms_controls_focus_q4/"
    "summary_matched_spatial_control_sweep_full"
)

REGIONS = ("central", "frontal", "parietal", "occipital", "temporal")
WINDOWS = (
    (0.0, 250.0),
    (125.0, 375.0),
    (250.0, 500.0),
)


def _holm(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    finite = ~np.isnan(pvals)
    out = np.full_like(pvals, np.nan)
    if not finite.any():
        return out
    p = pvals[finite]
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        v = (m - rank) * p[idx]
        running_max = max(running_max, v)
        adj[idx] = min(running_max, 1.0)
    out[finite] = adj
    return out


def main() -> int:
    warnings.filterwarnings("ignore")
    rows = []
    for region in REGIONS:
        for start_ms, end_ms in WINDOWS:
            label = f"{region}_{int(start_ms)}_{int(end_ms)}"
            csv = SWEEP_DIR / label / "matched_spatial_control_trial_scores.csv"
            if not csv.exists():
                print(f"SKIP missing {csv}")
                continue

            trial_scores = pd.read_csv(csv)
            try:
                frame = build_trial_level_frame(
                    trial_scores,
                    raw_model=DEFAULT_RAW_MODEL,
                    hybrid_model=DEFAULT_HYBRID_MODEL,
                    baseline_aug_model=DEFAULT_BASELINE_AUGMENTED_MODEL,
                    focus_start_ms=start_ms,
                    focus_end_ms=end_ms,
                )
            except Exception as exc:
                print(f"FAIL build {label}: {exc}")
                continue

            n_trials = int(len(frame))
            n_subjects = int(frame["subject"].nunique())
            n_hits = int((frame["sdt_int"] == 1).sum())

            try:
                det_lme = fit_detection_mixed_effects(frame)
                det_a = det_lme[
                    (det_lme["specification"] == "A_pred_fit_contrib_on_sdt")
                    & (det_lme["term"] == "sdt_int (hit=1)")
                ].iloc[0]
                det_beta = float(det_a["beta"])
                det_p = float(det_a["p"])
                det_z = float(det_a["z"])
                det_lo = float(det_a["ci95_low"])
                det_hi = float(det_a["ci95_high"])
            except Exception as exc:
                print(f"FAIL det LME {label}: {exc}")
                det_beta = det_p = det_z = det_lo = det_hi = np.nan

            try:
                conf_lme = fit_confidence_mixed_effects(frame)
                if conf_lme.empty:
                    raise ValueError("empty confidence LME table")
                conf_a = conf_lme[
                    (conf_lme["specification"] == "A_pred_fit_contrib_on_confidence")
                    & (conf_lme["term"] == "confidence")
                ].iloc[0]
                conf_beta = float(conf_a["beta"])
                conf_p = float(conf_a["p"])
                conf_z = float(conf_a["z"])
                conf_lo = float(conf_a["ci95_low"])
                conf_hi = float(conf_a["ci95_high"])
            except Exception as exc:
                print(f"FAIL conf LME {label}: {exc}")
                conf_beta = conf_p = conf_z = conf_lo = conf_hi = np.nan

            try:
                per_sub = per_subject_spearman_dissociation(frame)
                spear = summarise_spearman_dissociation(per_sub)
                rho_det = spear[spear["test"] == "detection_rho_vs_zero"].iloc[0]
                rho_con = spear[spear["test"] == "confidence_rho_vs_zero"].iloc[0]
                paired = spear[
                    spear["test"] == "detection_minus_confidence_rho_paired"
                ].iloc[0]
                rho_det_mean = float(rho_det["mean"])
                rho_det_p = float(rho_det["ttest_p"])
                rho_con_mean = float(rho_con["mean"])
                rho_con_p = float(rho_con["ttest_p"])
                rho_paired_mean = float(paired["mean"])
                rho_paired_p = float(paired["ttest_p"])
            except Exception as exc:
                print(f"FAIL spearman {label}: {exc}")
                rho_det_mean = rho_det_p = rho_con_mean = rho_con_p = np.nan
                rho_paired_mean = rho_paired_p = np.nan

            rows.append(
                {
                    "region": region,
                    "window_start_ms": start_ms,
                    "window_end_ms": end_ms,
                    "label": label,
                    "n_trials": n_trials,
                    "n_subjects": n_subjects,
                    "n_hits": n_hits,
                    "lme_det_beta": det_beta,
                    "lme_det_z": det_z,
                    "lme_det_p": det_p,
                    "lme_det_ci95_low": det_lo,
                    "lme_det_ci95_high": det_hi,
                    "lme_conf_beta": conf_beta,
                    "lme_conf_z": conf_z,
                    "lme_conf_p": conf_p,
                    "lme_conf_ci95_low": conf_lo,
                    "lme_conf_ci95_high": conf_hi,
                    "spearman_det_mean": rho_det_mean,
                    "spearman_det_ttest_p": rho_det_p,
                    "spearman_conf_mean": rho_con_mean,
                    "spearman_conf_ttest_p": rho_con_p,
                    "spearman_det_minus_conf_mean": rho_paired_mean,
                    "spearman_det_minus_conf_ttest_p": rho_paired_p,
                }
            )
            print(
                f"{label}: det beta={det_beta:+.4f} p={det_p:.2e}  "
                f"conf beta={conf_beta:+.4f} p={conf_p:.2e}  "
                f"n={n_trials}"
            )

    out_df = pd.DataFrame(rows)
    out_df["lme_det_p_holm"] = _holm(out_df["lme_det_p"].values)
    out_df["lme_conf_p_holm"] = _holm(out_df["lme_conf_p"].values)
    out_df["spearman_det_p_holm"] = _holm(out_df["spearman_det_ttest_p"].values)
    out_df["spearman_conf_p_holm"] = _holm(out_df["spearman_conf_ttest_p"].values)

    out_dir = SWEEP_DIR.parent / "summary_mechanistic_dissociation_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_table(out_df, out_dir / "mechanistic_sweep_table.csv")
    print(f"\nSaved {out_dir / 'mechanistic_sweep_table.csv'}")
    print("\n=== Detection LME (Holm-corrected across 15 region x window cells) ===")
    print(
        out_df[
            [
                "region",
                "window_start_ms",
                "window_end_ms",
                "lme_det_beta",
                "lme_det_p",
                "lme_det_p_holm",
                "lme_conf_beta",
                "lme_conf_p",
                "lme_conf_p_holm",
                "n_trials",
            ]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
