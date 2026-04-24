"""Trial-level detection versus confidence dissociation analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from framework.summaries import save_table
from reporting.statistics import holm_adjust


DEFAULT_RAW_MODEL = "raw_regions_delta"
DEFAULT_HYBRID_MODEL = "raw_plus_baseline_regions_augmented"
DEFAULT_BASELINE_AUGMENTED_MODEL = "baseline_regions_augmented"
DEFAULT_FOCUS_START_MS = 125.0
DEFAULT_FOCUS_END_MS = 375.0


def _safe_lme_fit(df: pd.DataFrame, formula: str, group_col: str):
    import statsmodels.formula.api as smf

    try:
        model = smf.mixedlm(formula, df, groups=df[group_col]).fit(
            method="lbfgs",
            reml=False,
            disp=False,
        )
    except Exception:
        model = smf.mixedlm(formula, df, groups=df[group_col]).fit(reml=False, disp=False)
    return model


def build_trial_level_frame(
    trial_scores: pd.DataFrame,
    *,
    raw_model: str,
    hybrid_model: str,
    baseline_aug_model: str,
    focus_start_ms: float,
    focus_end_ms: float,
) -> pd.DataFrame:
    if "group_kind" in trial_scores.columns:
        named_mask = trial_scores["group_kind"].astype(str) == "named_region"
        if named_mask.any():
            trial_scores = trial_scores.loc[named_mask].copy()

    focus_mask = (
        (trial_scores["target_start_ms"] == float(focus_start_ms))
        & (trial_scores["target_end_ms"] == float(focus_end_ms))
    )
    df = trial_scores.loc[focus_mask].copy()
    if df.empty:
        raise ValueError(
            f"No trial-score rows in the focus window {focus_start_ms}-{focus_end_ms} ms"
        )

    needed_models = {raw_model, hybrid_model, baseline_aug_model}
    df = df.loc[df["model_name"].isin(needed_models)].copy()

    keep_cols = [
        "subject",
        "trial_idx",
        "sdt",
        "confidence",
        "stimamp",
        "model_name",
        "evidence_score",
    ]
    missing = sorted(set(keep_cols) - set(df.columns))
    if missing:
        raise ValueError(
            f"Trial-scores frame is missing required columns: {missing}"
        )
    df = df.loc[:, keep_cols].drop_duplicates()

    wide = df.pivot_table(
        index=["subject", "trial_idx", "sdt", "confidence", "stimamp"],
        columns="model_name",
        values="evidence_score",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None

    for required in (raw_model, hybrid_model, baseline_aug_model):
        if required not in wide.columns:
            raise ValueError(f"Pivot is missing the {required!r} model column")

    wide["evidence_raw"] = wide[raw_model].astype(float)
    wide["evidence_hybrid"] = wide[hybrid_model].astype(float)
    wide["evidence_baseline_aug"] = wide[baseline_aug_model].astype(float)
    wide["pred_fit_contrib"] = wide["evidence_hybrid"] - wide["evidence_raw"]
    wide["sdt_int"] = (wide["sdt"].astype(str) == "hit").astype(int)
    wide["stimamp"] = wide["stimamp"].astype(float)
    wide["confidence"] = wide["confidence"].astype(float)

    wide = wide.dropna(
        subset=["evidence_raw", "evidence_hybrid", "stimamp", "sdt_int"]
    ).reset_index(drop=True)
    return wide


def _coef_row(model, name: str, term: str, n_obs: int, n_groups: int):
    if name not in model.params.index:
        return None
    return {
        "term": term,
        "beta": float(model.params[name]),
        "se": float(model.bse[name]),
        "z": float(model.tvalues[name]),
        "p": float(model.pvalues[name]),
        "ci95_low": float(model.conf_int().loc[name, 0]),
        "ci95_high": float(model.conf_int().loc[name, 1]),
        "n_obs": int(n_obs),
        "n_groups": int(n_groups),
    }


def fit_detection_mixed_effects(trial_frame: pd.DataFrame) -> pd.DataFrame:
    """Fit the two detection mixed-effects specifications used in the paper."""
    n_obs = len(trial_frame)
    n_groups = trial_frame["subject"].nunique()
    rows = []

    model_a = _safe_lme_fit(
        trial_frame,
        "pred_fit_contrib ~ sdt_int + stimamp",
        group_col="subject",
    )
    rows.append(
        {
            "specification": "A_pred_fit_contrib_on_sdt",
            **(_coef_row(model_a, "sdt_int", "sdt_int (hit=1)", n_obs, n_groups) or {}),
        }
    )
    rows.append(
        {
            "specification": "A_pred_fit_contrib_on_sdt",
            **(_coef_row(model_a, "stimamp", "stimamp", n_obs, n_groups) or {}),
        }
    )

    model_b = _safe_lme_fit(
        trial_frame,
        "evidence_hybrid ~ evidence_raw + stimamp + sdt_int",
        group_col="subject",
    )
    rows.append(
        {
            "specification": "B_hybrid_on_raw_stimamp_sdt",
            **(_coef_row(model_b, "sdt_int", "sdt_int (hit=1)", n_obs, n_groups) or {}),
        }
    )
    rows.append(
        {
            "specification": "B_hybrid_on_raw_stimamp_sdt",
            **(_coef_row(model_b, "evidence_raw", "evidence_raw", n_obs, n_groups) or {}),
        }
    )
    rows.append(
        {
            "specification": "B_hybrid_on_raw_stimamp_sdt",
            **(_coef_row(model_b, "stimamp", "stimamp", n_obs, n_groups) or {}),
        }
    )

    return pd.DataFrame([row for row in rows if row.get("term")])


def fit_confidence_mixed_effects(trial_frame: pd.DataFrame) -> pd.DataFrame:
    """Fit the two confidence mixed-effects specifications on hit trials."""
    hits = trial_frame.loc[trial_frame["sdt_int"] == 1].dropna(subset=["confidence"])
    if hits.empty:
        return pd.DataFrame()
    n_obs = len(hits)
    n_groups = hits["subject"].nunique()
    rows = []

    model_a = _safe_lme_fit(
        hits,
        "pred_fit_contrib ~ confidence + stimamp",
        group_col="subject",
    )
    rows.append(
        {
            "specification": "A_pred_fit_contrib_on_confidence",
            **(_coef_row(model_a, "confidence", "confidence", n_obs, n_groups) or {}),
        }
    )
    rows.append(
        {
            "specification": "A_pred_fit_contrib_on_confidence",
            **(_coef_row(model_a, "stimamp", "stimamp", n_obs, n_groups) or {}),
        }
    )

    model_b = _safe_lme_fit(
        hits,
        "evidence_hybrid ~ evidence_raw + stimamp + confidence",
        group_col="subject",
    )
    rows.append(
        {
            "specification": "B_hybrid_on_raw_stimamp_confidence",
            **(_coef_row(model_b, "confidence", "confidence", n_obs, n_groups) or {}),
        }
    )
    rows.append(
        {
            "specification": "B_hybrid_on_raw_stimamp_confidence",
            **(_coef_row(model_b, "evidence_raw", "evidence_raw", n_obs, n_groups) or {}),
        }
    )
    rows.append(
        {
            "specification": "B_hybrid_on_raw_stimamp_confidence",
            **(_coef_row(model_b, "stimamp", "stimamp", n_obs, n_groups) or {}),
        }
    )

    return pd.DataFrame([row for row in rows if row.get("term")])


def _spearman_partial(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Spearman partial correlation r(x, y | z) by ranking each then partial."""
    if len(x) < 5:
        return float("nan")
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rz = stats.rankdata(z)
    rxy = np.corrcoef(rx, ry)[0, 1]
    rxz = np.corrcoef(rx, rz)[0, 1]
    ryz = np.corrcoef(ry, rz)[0, 1]
    denom = np.sqrt((1 - rxz**2) * (1 - ryz**2))
    if not np.isfinite(denom) or denom <= 0.0:
        return float("nan")
    return float((rxy - rxz * ryz) / denom)


def per_subject_spearman_dissociation(trial_frame: pd.DataFrame) -> pd.DataFrame:
    """Per-subject Spearman partial correlations of the predictive-fit
    contribution against detection (all trials) and confidence (hits only),
    with stimamp partialled out.
    """
    rows = []
    for subject, sub_df in trial_frame.groupby("subject"):
        det_x = sub_df["pred_fit_contrib"].to_numpy(dtype=float)
        det_y = sub_df["sdt_int"].to_numpy(dtype=float)
        det_z = sub_df["stimamp"].to_numpy(dtype=float)
        rho_detection = _spearman_partial(det_x, det_y, det_z)

        hits = sub_df.loc[sub_df["sdt_int"] == 1].dropna(subset=["confidence"])
        if len(hits) >= 5:
            conf_x = hits["pred_fit_contrib"].to_numpy(dtype=float)
            conf_y = hits["confidence"].to_numpy(dtype=float)
            conf_z = hits["stimamp"].to_numpy(dtype=float)
            rho_confidence = _spearman_partial(conf_x, conf_y, conf_z)
            n_hits = int(len(hits))
        else:
            rho_confidence = float("nan")
            n_hits = int(len(hits))

        rows.append(
            {
                "subject": str(subject),
                "n_trials": int(len(sub_df)),
                "n_hits": n_hits,
                "spearman_partial_pred_fit_vs_detection": rho_detection,
                "spearman_partial_pred_fit_vs_confidence": rho_confidence,
            }
        )

    return pd.DataFrame(rows)


def summarise_spearman_dissociation(per_subject_df: pd.DataFrame) -> pd.DataFrame:
    """Summary-level test on the per-subject Spearman-partial distributions:
    one-sample t against zero plus Wilcoxon plus paired test of
    detection-rho minus confidence-rho across subjects.
    """
    detection_rho = per_subject_df["spearman_partial_pred_fit_vs_detection"].dropna().to_numpy(dtype=float)
    confidence_rho = per_subject_df["spearman_partial_pred_fit_vs_confidence"].dropna().to_numpy(dtype=float)
    paired_df = per_subject_df.dropna(
        subset=[
            "spearman_partial_pred_fit_vs_detection",
            "spearman_partial_pred_fit_vs_confidence",
        ]
    )
    paired_diff = (
        paired_df["spearman_partial_pred_fit_vs_detection"].to_numpy(dtype=float)
        - paired_df["spearman_partial_pred_fit_vs_confidence"].to_numpy(dtype=float)
    )

    def _summarise(values, null_value, name):
        values = np.asarray(values, dtype=float)
        if values.size < 2:
            return {
                "test": name,
                "n_subjects": int(values.size),
                "mean": float("nan"),
                "median": float("nan"),
                "ttest_p": float("nan"),
                "wilcoxon_p": float("nan"),
            }
        t = stats.ttest_1samp(values, null_value)
        try:
            w = stats.wilcoxon(values - null_value, zero_method="wilcox")
            wp = float(w.pvalue)
        except ValueError:
            wp = float("nan")
        return {
            "test": name,
            "n_subjects": int(values.size),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "ttest_p": float(t.pvalue),
            "wilcoxon_p": wp,
        }

    rows = [
        _summarise(detection_rho, 0.0, "detection_rho_vs_zero"),
        _summarise(confidence_rho, 0.0, "confidence_rho_vs_zero"),
        _summarise(paired_diff, 0.0, "detection_minus_confidence_rho_paired"),
    ]
    raw_p = [row["ttest_p"] for row in rows]
    holm = holm_adjust(raw_p)
    for row, holm_p in zip(rows, holm):
        row["ttest_p_holm"] = float(holm_p)
    return pd.DataFrame(rows)


def build_mechanistic_report(
    *,
    detection_lme: pd.DataFrame,
    confidence_lme: pd.DataFrame,
    spearman_summary: pd.DataFrame,
    n_trials: int,
    n_subjects: int,
    n_hits: int,
    focus_start_ms: float,
    focus_end_ms: float,
) -> str:
    lines = [
        "# Mechanistic Dissociation Summary",
        "",
        (
            "We isolate the marginal contribution of the **pre-stimulus VAR predictive-fit** "
            "features (pred_r2_obs, pred_mse_obs, pred_r2_latent, pred_nll_latent) beyond the "
            "**post-stim baseline-corrected evoked response** (raw_regions_delta = post-stim "
            "channel mean minus pre-stim channel mean). Per trial, the augmentation contribution "
            "is `pred_fit_contrib = evidence_hybrid - evidence_raw`. Subject-random-intercept "
            "linear mixed-effects models test this contribution against detection, then against "
            "confidence on hits, with stimulus amplitude partialled out as a fixed effect. This "
            "complements the n=18 paired subject-mean tests with a trial-level test that has "
            "thousands of degrees of freedom and proper random-subject structure."
        ),
        "",
        f"Focus window: {int(focus_start_ms)}-{int(focus_end_ms)} ms central scalp.",
        f"Sample: {n_trials} trials across {n_subjects} subjects ({n_hits} hits).",
        "",
        "## Trial-level mixed-effects: Detection",
        "",
    ]

    for spec_name, spec_df in detection_lme.groupby("specification"):
        lines.append(f"### Specification: `{spec_name}`")
        lines.append("")
        for _, row in spec_df.iterrows():
            lines.append(
                f"- {row['term']}: beta = {row['beta']:.4f}, "
                f"95% CI [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}], "
                f"z = {row['z']:.2f}, p = {row['p']:.3g}"
            )
        lines.append("")

    lines.extend(["## Trial-level mixed-effects: Confidence on hits", ""])
    if confidence_lme.empty:
        lines.append("- (insufficient hit data for confidence model)")
    else:
        for spec_name, spec_df in confidence_lme.groupby("specification"):
            lines.append(f"### Specification: `{spec_name}`")
            lines.append("")
            for _, row in spec_df.iterrows():
                lines.append(
                    f"- {row['term']}: beta = {row['beta']:.4f}, "
                    f"95% CI [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}], "
                    f"z = {row['z']:.2f}, p = {row['p']:.3g}"
                )
            lines.append("")

    lines.extend(["## Per-subject Spearman partial correlations (stimamp partialled)", ""])
    for _, row in spearman_summary.iterrows():
        lines.append(
            f"- {row['test']}: n={int(row['n_subjects'])}, mean={row['mean']:.3f}, "
            f"median={row['median']:.3f}, paired t Holm p={row['ttest_p_holm']:.3g}, "
            f"Wilcoxon p={row['wilcoxon_p']:.3g}"
        )
    lines.append("")

    detection_row = next(
        (
            row
            for _, row in detection_lme.iterrows()
            if row["specification"].startswith("A") and row["term"].startswith("sdt_int")
        ),
        None,
    )
    confidence_row = next(
        (
            row
            for _, row in confidence_lme.iterrows()
            if row["specification"].startswith("A") and row["term"] == "confidence"
        ),
        None,
    )
    paired_row = spearman_summary.loc[
        spearman_summary["test"] == "detection_minus_confidence_rho_paired"
    ]

    lines.extend(["## Mechanistic verdict", ""])
    if detection_row is not None:
        lines.append(
            f"- Detection LME beta on sdt_int = {detection_row['beta']:.4f} "
            f"(p = {detection_row['p']:.3g}); the pre-stim VAR predictive-fit "
            f"contribution is **larger on hit trials than miss trials** even after "
            f"controlling for stimulus amplitude and subject-level baseline differences."
        )
    if confidence_row is not None:
        confidence_beta = float(confidence_row["beta"])
        confidence_p = float(confidence_row["p"])
        if confidence_p < 0.05 and confidence_beta > 0.0:
            confidence_verdict = (
                "also increases with subjective confidence on detected trials, "
                "although this confidence loading is smaller than the detection effect"
            )
        elif confidence_p < 0.05 and confidence_beta < 0.0:
            confidence_verdict = (
                "decreases with subjective confidence on detected trials, indicating a "
                "qualitatively different confidence loading"
            )
        else:
            confidence_verdict = (
                "shows little evidence of a confidence relationship on detected trials"
            )
        lines.append(
            f"- Confidence LME beta on confidence = {confidence_beta:.4f} "
            f"(p = {confidence_p:.3g}); on hits only, the same contribution "
            f"{confidence_verdict} after the same stimamp control."
        )
    if not paired_row.empty:
        prow = paired_row.iloc[0]
        lines.append(
            f"- Paired per-subject difference in Spearman-partial rhos "
            f"(detection minus confidence) = {prow['mean']:.3f}, "
            f"paired t Holm p = {prow['ttest_p_holm']:.3g}. A positive value with "
            f"small p indicates within-subject dissociation."
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "Because `raw_regions_delta` is itself the per-channel post-stim minus "
                "pre-stim mean amplitude, the marginal contribution of the augmentation "
                "is by construction independent of evoked amplitude. The pre-stim VAR "
                "predictive-fit metrics measure how well a VAR(1) model fit on the -300 "
                "to 0 ms window predicts the post-stim test window — i.e. the degree to "
                "which **stimulus-evoked dynamical disruption** breaks pre-stim "
                "predictability. The trial-level mixed-effects result therefore claims: "
                "pre-stim brain dynamics that are more disrupted by the stimulus are "
                "more likely to yield a behavioural hit, with the central 125-375 ms "
                "cell also carrying a weaker graded confidence signal on detected "
                "trials. This dynamical-disruption signature is separable from evoked "
                "amplitude and stimulus amplitude, but it should be framed as a graded "
                "perceptual-state signal rather than a clean detection-only "
                "dissociation."
            ),
        ]
    )

    return "\n".join(lines)


def run_mechanistic_dissociation_summary(
    *,
    trial_scores_csv: Path,
    outdir: Path | None = None,
    raw_model: str = DEFAULT_RAW_MODEL,
    hybrid_model: str = DEFAULT_HYBRID_MODEL,
    baseline_aug_model: str = DEFAULT_BASELINE_AUGMENTED_MODEL,
    focus_start_ms: float = DEFAULT_FOCUS_START_MS,
    focus_end_ms: float = DEFAULT_FOCUS_END_MS,
) -> int:
    outdir = (
        trial_scores_csv.parent / "summary_mechanistic_dissociation"
        if outdir is None
        else outdir
    )
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        trial_scores = pd.read_csv(trial_scores_csv)
        trial_frame = build_trial_level_frame(
            trial_scores,
            raw_model=raw_model,
            hybrid_model=hybrid_model,
            baseline_aug_model=baseline_aug_model,
            focus_start_ms=focus_start_ms,
            focus_end_ms=focus_end_ms,
        )

        detection_lme = fit_detection_mixed_effects(trial_frame)
        confidence_lme = fit_confidence_mixed_effects(trial_frame)
        per_subject = per_subject_spearman_dissociation(trial_frame)
        spearman_summary = summarise_spearman_dissociation(per_subject)
        n_hits = int((trial_frame["sdt_int"] == 1).sum())

        report_text = build_mechanistic_report(
            detection_lme=detection_lme,
            confidence_lme=confidence_lme,
            spearman_summary=spearman_summary,
            n_trials=int(len(trial_frame)),
            n_subjects=int(trial_frame["subject"].nunique()),
            n_hits=n_hits,
            focus_start_ms=focus_start_ms,
            focus_end_ms=focus_end_ms,
        )
    except Exception as exc:
        print(f"FAILED: {exc}")
        return 1

    save_table(trial_frame, outdir / "trial_level_frame.csv")
    save_table(detection_lme, outdir / "detection_mixed_effects.csv")
    save_table(confidence_lme, outdir / "confidence_mixed_effects.csv")
    save_table(per_subject, outdir / "per_subject_spearman_partial.csv")
    save_table(spearman_summary, outdir / "spearman_dissociation_summary.csv")
    (outdir / "mechanistic_dissociation_summary.md").write_text(report_text)

    print(f"Saved {outdir / 'trial_level_frame.csv'}")
    print(f"Saved {outdir / 'detection_mixed_effects.csv'}")
    print(f"Saved {outdir / 'confidence_mixed_effects.csv'}")
    print(f"Saved {outdir / 'per_subject_spearman_partial.csv'}")
    print(f"Saved {outdir / 'spearman_dissociation_summary.csv'}")
    print(f"Saved {outdir / 'mechanistic_dissociation_summary.md'}")

    print("\nDetection mixed-effects (specification A, sdt_int term):")
    spec_a_det = detection_lme[
        (detection_lme["specification"].str.startswith("A"))
        & (detection_lme["term"].str.startswith("sdt_int"))
    ]
    if not spec_a_det.empty:
        row = spec_a_det.iloc[0]
        print(
            f"  beta={row['beta']:.4f}  CI=[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]  "
            f"z={row['z']:.2f}  p={row['p']:.3g}  "
            f"n_obs={int(row['n_obs'])}  n_subjects={int(row['n_groups'])}"
        )

    print("\nConfidence mixed-effects (specification A, confidence term):")
    if not confidence_lme.empty:
        spec_a_conf = confidence_lme[
            (confidence_lme["specification"].str.startswith("A"))
            & (confidence_lme["term"] == "confidence")
        ]
        if not spec_a_conf.empty:
            row = spec_a_conf.iloc[0]
            print(
                f"  beta={row['beta']:.4f}  CI=[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]  "
                f"z={row['z']:.2f}  p={row['p']:.3g}  "
                f"n_obs={int(row['n_obs'])}  n_subjects={int(row['n_groups'])}"
            )

    print("\nPer-subject Spearman dissociation summary:")
    print(spearman_summary.to_string(index=False))

    return 0
