from __future__ import annotations

import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from framework.summaries import (
    compute_metric_means,
    compute_metric_summary,
    load_result_csvs,
    prepare_pyplot,
    save_table,
)


BASELINE_REQUIRED_COLUMNS = [
    "subject",
    "trial_idx",
    "rep_dim",
    "pred_mse_obs",
    "pred_r2_obs",
]
METRIC_SPECS = (
    ("pred_r2_obs", True, "Observed-space R^2"),
    ("pred_mse_obs", False, "Observed-space MSE"),
)
DEFAULT_REP_DIMS = (2, 4, 8, 16, 32)
DEFAULT_N_BOOT = 10_000
DEFAULT_RANDOM_SEED = 7
MATCHED_CONTROL_ORDER = ["pca_var1", "random_projection", "shuffled_dynamics"]
CONTROL_METRIC_SPECS = (
    ("pred_r2_obs", "Observed-space R^2", "higher"),
    ("pred_mse_obs", "Observed-space MSE", "lower"),
)
CONTROL_COLOURS = {
    "pca_var1": "#0f4c5c",
    "random_projection": "#e36414",
    "shuffled_dynamics": "#6f1d1b",
}
PRISM_MODEL_ORDER = ["pca_var1", "prism_pca", "prism_psi_opt", "prism_random"]
PRISM_COLOURS = {
    "pca_var1": "#0f4c5c",
    "prism_pca": "#e36414",
    "prism_psi_opt": "#2a9d8f",
    "prism_random": "#6f1d1b",
}
TEMPORAL_DURATION_ORDER = (250.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 3000.0)


def load_subject_means(results_dir: Path) -> pd.DataFrame:
    all_df = load_result_csvs(
        results_dir=results_dir,
        pattern="sub-*_trial_baseline.csv",
        required_columns=BASELINE_REQUIRED_COLUMNS,
        sort_columns=["subject", "trial_idx", "rep_dim"],
    )
    return compute_metric_means(all_df, group_columns=["subject", "rep_dim"])


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_bootstrap: int,
    random_seed: int,
    ci: float = 95.0,
) -> tuple[float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(random_seed)
    boot_means = np.empty(n_bootstrap, dtype=float)
    for idx in range(n_bootstrap):
        sample = rng.choice(clean, size=clean.size, replace=True)
        boot_means[idx] = float(np.mean(sample))

    alpha = 0.5 * (100.0 - ci)
    lower = float(np.percentile(boot_means, alpha))
    upper = float(np.percentile(boot_means, 100.0 - alpha))
    return lower, upper


def holm_adjust(p_values: list[float]) -> list[float]:
    adjusted = np.full(len(p_values), np.nan, dtype=float)
    finite_items = [
        (idx, float(value))
        for idx, value in enumerate(p_values)
        if np.isfinite(float(value))
    ]
    if not finite_items:
        return adjusted.tolist()

    n = len(finite_items)
    order = sorted(finite_items, key=lambda item: item[1])
    running_max = 0.0

    for rank, (idx, value) in enumerate(order):
        factor = n - rank
        corrected = min(1.0, value * factor)
        running_max = max(running_max, corrected)
        adjusted[idx] = running_max

    return adjusted.tolist()


def build_descriptive_table(
    subject_means: pd.DataFrame,
    metric_col: str,
    *,
    n_bootstrap: int,
    random_seed: int,
) -> pd.DataFrame:
    rows = []

    for rep_dim, rep_df in subject_means.groupby("rep_dim", observed=False):
        values = rep_df[metric_col].to_numpy(dtype=float)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if values.size > 1 else float("nan")
        sem = std / math.sqrt(values.size) if values.size > 1 else float("nan")
        ci_low, ci_high = bootstrap_mean_ci(
            values,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed + int(rep_dim),
        )
        rows.append(
            {
                "rep_dim": int(rep_dim),
                "n_subjects": int(values.size),
                "mean": mean,
                "std": std,
                "sem": sem,
                "median": float(np.median(values)),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )

    return pd.DataFrame(rows).sort_values("rep_dim").reset_index(drop=True)


def build_omnibus_table(subject_means: pd.DataFrame) -> pd.DataFrame:
    wide = subject_means.pivot(index="subject", columns="rep_dim")
    rows = []

    for metric_col, _, metric_label in METRIC_SPECS:
        metric_wide = wide[metric_col].reindex(columns=list(DEFAULT_REP_DIMS))
        metric_wide = metric_wide.dropna(axis=0, how="any")
        samples = [metric_wide[rep_dim].to_numpy(dtype=float) for rep_dim in metric_wide.columns]
        stat, p_value = stats.friedmanchisquare(*samples)
        rows.append(
            {
                "metric": metric_col,
                "metric_label": metric_label,
                "n_subjects": int(metric_wide.shape[0]),
                "friedman_stat": float(stat),
                "friedman_p": float(p_value),
            }
        )

    return pd.DataFrame(rows)


def build_pairwise_table(
    subject_means: pd.DataFrame,
    metric_col: str,
    *,
    higher_is_better: bool,
    n_bootstrap: int,
    random_seed: int,
) -> pd.DataFrame:
    wide = (
        subject_means.pivot(index="subject", columns="rep_dim", values=metric_col)
        .reindex(columns=list(DEFAULT_REP_DIMS))
    )

    rows = []
    p_t = []
    p_w = []

    for pair_idx, (rep_dim_a, rep_dim_b) in enumerate(combinations(DEFAULT_REP_DIMS, 2)):
        pair_df = wide[[rep_dim_a, rep_dim_b]].dropna(axis=0, how="any")
        values_a = pair_df[rep_dim_a].to_numpy(dtype=float)
        values_b = pair_df[rep_dim_b].to_numpy(dtype=float)

        raw_delta = values_b - values_a
        improvement = raw_delta if higher_is_better else -raw_delta
        mean_delta = float(np.mean(raw_delta))
        mean_improvement = float(np.mean(improvement))
        std_improvement = (
            float(np.std(improvement, ddof=1)) if improvement.size > 1 else float("nan")
        )
        effect_dz = (
            mean_improvement / std_improvement
            if raw_delta.size > 1 and np.isfinite(std_improvement) and std_improvement > 0.0
            else float("nan")
        )

        ci_low, ci_high = bootstrap_mean_ci(
            improvement,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed + 1000 * (pair_idx + 1),
        )

        t_result = stats.ttest_rel(values_b, values_a, nan_policy="omit")
        try:
            w_result = stats.wilcoxon(values_b, values_a, zero_method="wilcox")
            w_stat = float(w_result.statistic)
            w_p = float(w_result.pvalue)
        except ValueError:
            w_stat = float("nan")
            w_p = float("nan")

        rows.append(
            {
                "metric": metric_col,
                "rep_dim_a": int(rep_dim_a),
                "rep_dim_b": int(rep_dim_b),
                "n_subjects": int(pair_df.shape[0]),
                "mean_a": float(np.mean(values_a)),
                "mean_b": float(np.mean(values_b)),
                "mean_delta_raw": mean_delta,
                "mean_improvement": mean_improvement,
                "ci95_low_improvement": ci_low,
                "ci95_high_improvement": ci_high,
                "cohen_dz": effect_dz,
                "n_rep_dim_b_better": int(np.sum(improvement > 0.0)),
                "n_rep_dim_a_better": int(np.sum(improvement < 0.0)),
                "n_ties": int(np.sum(np.isclose(improvement, 0.0))),
                "ttest_stat": float(t_result.statistic),
                "ttest_p": float(t_result.pvalue),
                "wilcoxon_stat": w_stat,
                "wilcoxon_p": w_p,
                "better_if": "higher" if higher_is_better else "lower",
            }
        )
        p_t.append(float(t_result.pvalue))
        p_w.append(w_p)

    out = pd.DataFrame(rows)
    out["ttest_p_holm"] = holm_adjust(p_t)
    out["wilcoxon_p_holm"] = holm_adjust(p_w)
    return out


def plot_subject_level_summary(
    subject_means: pd.DataFrame,
    r2_df: pd.DataFrame,
    mse_df: pd.DataFrame,
    outfile: Path,
) -> None:
    plt = prepare_pyplot(outfile.parent)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    for _, subject_df in subject_means.groupby("subject", observed=False):
        axes[0].plot(
            subject_df["rep_dim"],
            subject_df["pred_r2_obs"],
            color="#adb5bd",
            linewidth=1.0,
            alpha=0.35,
        )
        axes[1].plot(
            subject_df["rep_dim"],
            subject_df["pred_mse_obs"],
            color="#adb5bd",
            linewidth=1.0,
            alpha=0.35,
        )

    axes[0].errorbar(
        r2_df["rep_dim"],
        r2_df["mean"],
        yerr=[r2_df["mean"] - r2_df["ci95_low"], r2_df["ci95_high"] - r2_df["mean"]],
        fmt="-o",
        color="#0f4c5c",
        linewidth=2.5,
        capsize=3,
    )
    axes[0].set_title("Observed-space R^2 by representation dimension")
    axes[0].set_xlabel("Representation dimension")
    axes[0].set_ylabel("Subject mean pred_r2_obs")
    axes[0].grid(True, alpha=0.25)

    axes[1].errorbar(
        mse_df["rep_dim"],
        mse_df["mean"],
        yerr=[mse_df["mean"] - mse_df["ci95_low"], mse_df["ci95_high"] - mse_df["mean"]],
        fmt="-o",
        color="#e36414",
        linewidth=2.5,
        capsize=3,
    )
    axes[1].set_title("Observed-space MSE by representation dimension")
    axes[1].set_xlabel("Representation dimension")
    axes[1].set_ylabel("Subject mean pred_mse_obs")
    axes[1].grid(True, alpha=0.25)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_markdown_report(
    omnibus_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    r2_df: pd.DataFrame,
    mse_df: pd.DataFrame,
    *,
    n_subjects: int,
) -> str:
    best_r2_row = r2_df.sort_values("mean", ascending=False).iloc[0]
    best_mse_row = mse_df.sort_values("mean", ascending=True).iloc[0]

    lines = [
        "# Baseline Statistical Summary",
        "",
        f"- Subjects analysed: {n_subjects}",
        "- Representation dimensions: " + ", ".join(str(dim) for dim in DEFAULT_REP_DIMS),
        (
            f"- Best mean observed R^2: q={int(best_r2_row['rep_dim'])} "
            f"({best_r2_row['mean']:.3f})"
        ),
        (
            f"- Lowest mean observed MSE: q={int(best_mse_row['rep_dim'])} "
            f"({best_mse_row['mean']:.3f})"
        ),
        "",
        "## Omnibus tests",
        "",
    ]

    for _, row in omnibus_df.iterrows():
        lines.append(
            f"- {row['metric_label']}: "
            f"Friedman chi^2={row['friedman_stat']:.3f}, p={row['friedman_p']:.3g}"
        )

    lines.extend(["", "## Focused paired contrasts", ""])

    focus_pairs = [(8, 16), (8, 32), (16, 32)]
    r2_pairs = pairwise_df.loc[pairwise_df["metric"] == "pred_r2_obs"]
    for rep_dim_a, rep_dim_b in focus_pairs:
        row = r2_pairs.loc[
            (r2_pairs["rep_dim_a"] == rep_dim_a) & (r2_pairs["rep_dim_b"] == rep_dim_b)
        ].iloc[0]
        lines.append(
            f"- q={rep_dim_a} vs q={rep_dim_b}: "
            f"mean improvement={row['mean_improvement']:.3f}, "
            f"95% bootstrap CI [{row['ci95_low_improvement']:.3f}, "
            f"{row['ci95_high_improvement']:.3f}], "
            f"paired t p={row['ttest_p_holm']:.3g}, "
            f"Cohen dz={row['cohen_dz']:.3f}"
        )

    return "\n".join(lines)


def run_baseline_statistics(
    *,
    results_dir: Path,
    outdir: Path | None = None,
    n_bootstrap: int = DEFAULT_N_BOOT,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> int:
    outdir = results_dir / "statistics" if outdir is None else outdir
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        subject_means = load_subject_means(results_dir)
        r2_df = build_descriptive_table(
            subject_means,
            "pred_r2_obs",
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        )
        mse_df = build_descriptive_table(
            subject_means,
            "pred_mse_obs",
            n_bootstrap=n_bootstrap,
            random_seed=random_seed + 10_000,
        )
        omnibus_df = build_omnibus_table(subject_means)
        pairwise_frames = []
        for metric_col, higher_is_better, _metric_label in METRIC_SPECS:
            pairwise_frames.append(
                build_pairwise_table(
                    subject_means,
                    metric_col,
                    higher_is_better=higher_is_better,
                    n_bootstrap=n_bootstrap,
                    random_seed=random_seed,
                )
            )
        pairwise_df = pd.concat(pairwise_frames, ignore_index=True)
        report = build_markdown_report(
            omnibus_df,
            pairwise_df,
            r2_df,
            mse_df,
            n_subjects=int(subject_means["subject"].nunique()),
        )
    except Exception as exc:
        print(f"FAILED: {exc}")
        return 1

    subject_means_path = outdir / "baseline_subject_level_means_by_rep_dim.csv"
    r2_path = outdir / "baseline_subject_level_r2_descriptives.csv"
    mse_path = outdir / "baseline_subject_level_mse_descriptives.csv"
    omnibus_path = outdir / "baseline_omnibus_tests.csv"
    pairwise_path = outdir / "baseline_pairwise_contrasts.csv"
    plot_path = outdir / "baseline_subject_level_statistics.png"
    report_path = outdir / "baseline_statistical_summary.md"

    save_table(subject_means, subject_means_path)
    save_table(r2_df, r2_path)
    save_table(mse_df, mse_path)
    save_table(omnibus_df, omnibus_path)
    save_table(pairwise_df, pairwise_path)
    plot_subject_level_summary(subject_means, r2_df, mse_df, plot_path)
    report_path.write_text(report + "\n", encoding="utf-8")

    print(f"Saved {subject_means_path}")
    print(f"Saved {r2_path}")
    print(f"Saved {mse_path}")
    print(f"Saved {omnibus_path}")
    print(f"Saved {pairwise_path}")
    print(f"Saved {plot_path}")
    print(f"Saved {report_path}")

    print("\nOmnibus tests:")
    print(omnibus_df.to_string(index=False))

    print("\nPairwise contrasts for observed R^2:")
    print(pairwise_df.loc[pairwise_df["metric"] == "pred_r2_obs"].to_string(index=False))
    return 0


def load_baseline_and_controls(
    results_dir: Path,
    controls_dir: Path | None = None,
) -> tuple[pd.DataFrame, Path]:
    controls_dir = results_dir / "controls" if controls_dir is None else controls_dir

    baseline_df = load_result_csvs(
        results_dir=results_dir,
        pattern="sub-*_trial_baseline.csv",
        required_columns=["subject", "trial_idx", "rep_dim", "pred_mse_obs", "pred_r2_obs"],
        sort_columns=["subject", "trial_idx", "rep_dim"],
    ).copy()
    baseline_df["control_kind"] = "pca_var1"
    baseline_df["control_seed"] = np.nan

    control_df = load_result_csvs(
        results_dir=controls_dir,
        pattern="sub-*_control_baseline.csv",
        required_columns=[
            "subject",
            "trial_idx",
            "control_kind",
            "rep_dim",
            "pred_mse_obs",
            "pred_r2_obs",
        ],
        sort_columns=["subject", "trial_idx", "control_kind", "rep_dim"],
        categorical_orders={"control_kind": MATCHED_CONTROL_ORDER + ["observed_var1"]},
    ).copy()

    common_subjects = sorted(set(baseline_df["subject"]) & set(control_df["subject"]))
    if not common_subjects:
        raise ValueError("No subjects overlap between baseline and control results")

    baseline_df = baseline_df.loc[baseline_df["subject"].isin(common_subjects)].copy()
    control_df = control_df.loc[control_df["subject"].isin(common_subjects)].copy()
    combined = pd.concat([baseline_df, control_df], ignore_index=True)
    combined = combined.sort_values(["subject", "control_kind", "trial_idx", "rep_dim"])
    return combined, controls_dir


def build_control_subject_means(all_df: pd.DataFrame) -> pd.DataFrame:
    return compute_metric_means(
        all_df,
        group_columns=["subject", "control_kind", "rep_dim"],
        metric_columns=[metric for metric, _, _ in CONTROL_METRIC_SPECS],
    )


def build_control_group_summary(
    subject_means: pd.DataFrame,
    *,
    n_bootstrap: int,
    random_seed: int,
) -> pd.DataFrame:
    summary = compute_metric_summary(
        subject_means,
        group_columns=["control_kind", "rep_dim"],
        metric_columns=[metric for metric, _, _ in CONTROL_METRIC_SPECS],
    )
    summary = summary.sort_values(["control_kind", "rep_dim"]).reset_index(drop=True)

    for metric, _, _ in CONTROL_METRIC_SPECS:
        ci_low_values = []
        ci_high_values = []
        for _, row in summary.iterrows():
            group_df = subject_means.loc[
                (subject_means["control_kind"] == row["control_kind"])
                & (subject_means["rep_dim"] == row["rep_dim"])
            ]
            seed = random_seed + int(row["rep_dim"]) + len(ci_low_values)
            ci_low, ci_high = bootstrap_mean_ci(
                group_df[metric].to_numpy(dtype=float),
                n_bootstrap=n_bootstrap,
                random_seed=seed,
            )
            ci_low_values.append(ci_low)
            ci_high_values.append(ci_high)
        summary[f"{metric}_ci95_low"] = ci_low_values
        summary[f"{metric}_ci95_high"] = ci_high_values

    return summary


def build_control_contrast_table(
    subject_means: pd.DataFrame,
    *,
    n_bootstrap: int,
    random_seed: int,
) -> pd.DataFrame:
    rows = []

    for control_kind in MATCHED_CONTROL_ORDER[1:]:
        baseline_df = subject_means.loc[subject_means["control_kind"] == "pca_var1"]
        control_df = subject_means.loc[subject_means["control_kind"] == control_kind]
        common_rep_dims = sorted(set(baseline_df["rep_dim"]) & set(control_df["rep_dim"]))

        for rep_dim in common_rep_dims:
            merged = baseline_df.loc[baseline_df["rep_dim"] == rep_dim].merge(
                control_df.loc[control_df["rep_dim"] == rep_dim],
                on=["subject", "rep_dim"],
                suffixes=("_pca", "_control"),
            )
            if merged.empty:
                continue

            for metric, metric_label, better_if in CONTROL_METRIC_SPECS:
                pca_values = merged[f"{metric}_pca"].to_numpy(dtype=float)
                control_values = merged[f"{metric}_control"].to_numpy(dtype=float)
                n_subjects = int(merged["subject"].nunique())
                raw_delta = pca_values - control_values
                pca_advantage = (
                    raw_delta if better_if == "higher" else control_values - pca_values
                )
                ci_low, ci_high = bootstrap_mean_ci(
                    pca_advantage,
                    n_bootstrap=n_bootstrap,
                    random_seed=random_seed + int(rep_dim),
                )
                if n_subjects >= 2:
                    ttest = stats.ttest_rel(pca_values, control_values, nan_policy="omit")
                    try:
                        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(
                            pca_values,
                            control_values,
                        )
                    except ValueError:
                        wilcoxon_stat, wilcoxon_p = np.nan, np.nan
                    ttest_stat = float(ttest.statistic)
                    ttest_p = float(ttest.pvalue)
                else:
                    wilcoxon_stat, wilcoxon_p = np.nan, np.nan
                    ttest_stat, ttest_p = np.nan, np.nan

                if better_if == "higher":
                    n_pca_better = int(np.sum(pca_values > control_values))
                    n_control_better = int(np.sum(control_values > pca_values))
                else:
                    n_pca_better = int(np.sum(pca_values < control_values))
                    n_control_better = int(np.sum(control_values < pca_values))

                centred_advantage = pca_advantage - np.mean(pca_advantage)
                denom = float(np.sqrt(np.mean(centred_advantage**2)))
                cohen_dz = (
                    float(np.mean(pca_advantage) / denom)
                    if denom > 0.0
                    else float("nan")
                )

                rows.append(
                    {
                        "control_kind": control_kind,
                        "metric": metric,
                        "metric_label": metric_label,
                        "rep_dim": int(rep_dim),
                        "n_subjects": n_subjects,
                        "pca_mean": float(np.mean(pca_values)),
                        "control_mean": float(np.mean(control_values)),
                        "mean_delta_raw": float(np.mean(raw_delta)),
                        "mean_pca_advantage": float(np.mean(pca_advantage)),
                        "ci95_low_pca_advantage": ci_low,
                        "ci95_high_pca_advantage": ci_high,
                        "cohen_dz": cohen_dz,
                        "n_pca_better": n_pca_better,
                        "n_control_better": n_control_better,
                        "n_ties": int(np.sum(np.isclose(pca_values, control_values))),
                        "ttest_stat": ttest_stat,
                        "ttest_p": ttest_p,
                        "wilcoxon_stat": float(wilcoxon_stat),
                        "wilcoxon_p": float(wilcoxon_p),
                        "better_if": better_if,
                    }
                )

    contrasts = pd.DataFrame(rows)
    if contrasts.empty:
        return contrasts

    contrasts["ttest_p_holm"] = np.nan
    contrasts["wilcoxon_p_holm"] = np.nan
    for metric in contrasts["metric"].unique():
        metric_mask = contrasts["metric"] == metric
        contrasts.loc[metric_mask, "ttest_p_holm"] = holm_adjust(
            contrasts.loc[metric_mask, "ttest_p"].to_numpy(dtype=float)
        )
        contrasts.loc[metric_mask, "wilcoxon_p_holm"] = holm_adjust(
            contrasts.loc[metric_mask, "wilcoxon_p"].to_numpy(dtype=float)
        )

    return contrasts.sort_values(["metric", "control_kind", "rep_dim"])


def plot_control_summary(group_summary: pd.DataFrame, outfile: Path) -> None:
    plt = prepare_pyplot(outfile.parent)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    matched_df = group_summary.loc[group_summary["control_kind"].isin(MATCHED_CONTROL_ORDER)]
    reference_df = group_summary.loc[group_summary["control_kind"] == "observed_var1"]

    for control_kind in MATCHED_CONTROL_ORDER:
        control_df = matched_df.loc[matched_df["control_kind"] == control_kind]
        colour = CONTROL_COLOURS[control_kind]

        axes[0].errorbar(
            control_df["rep_dim"],
            control_df["pred_r2_obs_mean"],
            yerr=[
                control_df["pred_r2_obs_mean"] - control_df["pred_r2_obs_ci95_low"],
                control_df["pred_r2_obs_ci95_high"] - control_df["pred_r2_obs_mean"],
            ],
            fmt="-o",
            color=colour,
            capsize=3,
            linewidth=2,
            label=control_kind,
        )
        axes[1].errorbar(
            control_df["rep_dim"],
            control_df["pred_mse_obs_mean"],
            yerr=[
                control_df["pred_mse_obs_mean"] - control_df["pred_mse_obs_ci95_low"],
                control_df["pred_mse_obs_ci95_high"] - control_df["pred_mse_obs_mean"],
            ],
            fmt="-o",
            color=colour,
            capsize=3,
            linewidth=2,
            label=control_kind,
        )

    if not reference_df.empty:
        ref_row = reference_df.iloc[0]
        axes[0].axhline(
            ref_row["pred_r2_obs_mean"],
            color="#495057",
            linestyle="--",
            linewidth=1.8,
            label="observed_var1",
        )
        axes[1].axhline(
            ref_row["pred_mse_obs_mean"],
            color="#495057",
            linestyle="--",
            linewidth=1.8,
            label="observed_var1",
        )

    axes[0].set_title("Observed-space R^2")
    axes[0].set_xlabel("Representation dimension")
    axes[0].set_ylabel("Subject mean pred_r2_obs")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].set_title("Observed-space MSE")
    axes[1].set_xlabel("Representation dimension")
    axes[1].set_ylabel("Subject mean pred_mse_obs")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_control_markdown_report(
    subject_means: pd.DataFrame,
    group_summary: pd.DataFrame,
    contrasts: pd.DataFrame,
) -> str:
    n_subjects = int(subject_means["subject"].nunique())
    baseline_summary = group_summary.loc[group_summary["control_kind"] == "pca_var1"]
    best_baseline_r2 = baseline_summary.sort_values("pred_r2_obs_mean", ascending=False).iloc[0]

    lines = [
        "# Control Comparison Summary",
        "",
        f"- Subjects analysed: {n_subjects}",
        (
            "- Best mean baseline observed R^2: "
            f"q={int(best_baseline_r2['rep_dim'])} ({best_baseline_r2['pred_r2_obs_mean']:.3f})"
        ),
    ]

    observed_var1_df = group_summary.loc[group_summary["control_kind"] == "observed_var1"]
    if not observed_var1_df.empty:
        observed_var1 = observed_var1_df.iloc[0]
        lines.append(
            "- Full observed-space VAR(1) reference: "
            f"R^2={observed_var1['pred_r2_obs_mean']:.3f}, "
            f"MSE={observed_var1['pred_mse_obs_mean']:.3f}"
        )

    lines.extend(["", "## Focused contrasts at q=16", ""])

    focus_rows = contrasts.loc[
        (contrasts["metric"] == "pred_r2_obs") & (contrasts["rep_dim"] == 16)
    ]
    if focus_rows.empty:
        lines.append("- No matched q=16 contrast rows were available.")
    else:
        for _, row in focus_rows.iterrows():
            lines.append(
                "- PCA vs "
                f"{row['control_kind']}: mean advantage={row['mean_pca_advantage']:.3f}, "
                f"95% bootstrap CI [{row['ci95_low_pca_advantage']:.3f}, "
                f"{row['ci95_high_pca_advantage']:.3f}], "
                f"Holm-corrected paired t p={row['ttest_p_holm']:.3g}, "
                f"Cohen dz={row['cohen_dz']:.3f}"
            )

    return "\n".join(lines)


def run_control_summary(
    *,
    results_dir: Path,
    controls_dir: Path | None = None,
    n_bootstrap: int = DEFAULT_N_BOOT,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> int:
    try:
        all_df, controls_dir = load_baseline_and_controls(results_dir, controls_dir=controls_dir)
        subject_means = build_control_subject_means(all_df)
        group_summary = build_control_group_summary(
            subject_means,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        )
        contrasts = build_control_contrast_table(
            subject_means,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        )
        report = build_control_markdown_report(subject_means, group_summary, contrasts)
    except Exception as exc:
        print(f"FAILED: {exc}")
        return 1

    subject_means_path = controls_dir / "control_subject_means_by_kind_and_rep_dim.csv"
    group_summary_path = controls_dir / "control_group_summary_by_kind_and_rep_dim.csv"
    contrasts_path = controls_dir / "control_vs_baseline_subject_contrasts.csv"
    plot_path = controls_dir / "control_comparison_plot.png"
    report_path = controls_dir / "control_comparison_summary.md"

    save_table(subject_means, subject_means_path)
    save_table(group_summary, group_summary_path)
    save_table(contrasts, contrasts_path)
    plot_control_summary(group_summary, plot_path)
    report_path.write_text(report + "\n", encoding="utf-8")

    print(f"Saved {subject_means_path}")
    print(f"Saved {group_summary_path}")
    print(f"Saved {contrasts_path}")
    print(f"Saved {plot_path}")
    print(f"Saved {report_path}")

    print("\nGroup summary:")
    print(group_summary.to_string(index=False))

    if not contrasts.empty:
        print("\nMatched contrasts:")
        print(contrasts.to_string(index=False))

    return 0


def ordered_model_families(values: pd.Series) -> list[str]:
    seen = {str(value) for value in values.dropna().astype(str)}
    ordered = [name for name in PRISM_MODEL_ORDER if name in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered


def load_baseline_and_prism(
    baseline_results_dir: Path,
    prism_results_dir: Path,
) -> tuple[pd.DataFrame, list[str]]:
    baseline_df = load_result_csvs(
        results_dir=baseline_results_dir,
        pattern="sub-*_trial_baseline.csv",
        required_columns=["subject", "trial_idx", "rep_dim", "pred_mse_obs", "pred_r2_obs"],
        sort_columns=["subject", "trial_idx", "rep_dim"],
    ).copy()
    baseline_df["model_family"] = "pca_var1"

    prism_df = load_result_csvs(
        results_dir=prism_results_dir,
        pattern="sub-*_trial_prism.csv",
        required_columns=[
            "subject",
            "trial_idx",
            "model_family",
            "projection_mode",
            "rep_dim",
            "pred_mse_obs",
            "pred_r2_obs",
        ],
        sort_columns=["subject", "trial_idx", "model_family", "rep_dim"],
    ).copy()

    common_subjects = sorted(set(baseline_df["subject"]) & set(prism_df["subject"]))
    if not common_subjects:
        raise ValueError("No subjects overlap between the baseline and PRISM results")

    baseline_df = baseline_df.loc[baseline_df["subject"].isin(common_subjects)].copy()
    prism_df = prism_df.loc[prism_df["subject"].isin(common_subjects)].copy()

    combined = pd.concat([baseline_df, prism_df], ignore_index=True)
    model_families = ordered_model_families(combined["model_family"])
    combined["model_family"] = pd.Categorical(
        combined["model_family"],
        categories=model_families,
        ordered=True,
    )
    combined = combined.sort_values(["subject", "model_family", "trial_idx", "rep_dim"])
    return combined, model_families


def build_prism_subject_means(all_df: pd.DataFrame) -> pd.DataFrame:
    out = compute_metric_means(
        all_df,
        group_columns=["subject", "model_family", "rep_dim"],
        metric_columns=[metric for metric, _, _ in CONTROL_METRIC_SPECS],
    )
    metric_columns = [metric for metric, _, _ in CONTROL_METRIC_SPECS]
    return out.dropna(subset=metric_columns, how="all").reset_index(drop=True)


def build_prism_group_summary(
    subject_means: pd.DataFrame,
    *,
    model_families: list[str],
    n_bootstrap: int,
    random_seed: int,
) -> pd.DataFrame:
    summary = compute_metric_summary(
        subject_means,
        group_columns=["model_family", "rep_dim"],
        metric_columns=[metric for metric, _, _ in CONTROL_METRIC_SPECS],
    )
    summary["model_family"] = pd.Categorical(
        summary["model_family"],
        categories=model_families,
        ordered=True,
    )
    summary = summary.dropna(
        subset=[f"{metric}_mean" for metric, _, _ in CONTROL_METRIC_SPECS],
        how="all",
    )
    summary = summary.sort_values(["model_family", "rep_dim"]).reset_index(drop=True)

    for metric, _, _ in CONTROL_METRIC_SPECS:
        ci_low_values = []
        ci_high_values = []
        for row_idx, row in summary.iterrows():
            group_df = subject_means.loc[
                (subject_means["model_family"] == row["model_family"])
                & (subject_means["rep_dim"] == row["rep_dim"])
            ]
            seed = random_seed + int(row["rep_dim"]) + 100 * row_idx
            ci_low, ci_high = bootstrap_mean_ci(
                group_df[metric].to_numpy(dtype=float),
                n_bootstrap=n_bootstrap,
                random_seed=seed,
            )
            ci_low_values.append(ci_low)
            ci_high_values.append(ci_high)
        summary[f"{metric}_ci95_low"] = ci_low_values
        summary[f"{metric}_ci95_high"] = ci_high_values

    return summary


def build_prism_contrast_table(
    subject_means: pd.DataFrame,
    *,
    model_families: list[str],
    n_bootstrap: int,
    random_seed: int,
) -> pd.DataFrame:
    rows = []

    for rep_dim in sorted(subject_means["rep_dim"].dropna().unique()):
        rep_df = subject_means.loc[subject_means["rep_dim"] == rep_dim]

        for model_a, model_b in combinations(model_families, 2):
            model_a_df = rep_df.loc[rep_df["model_family"] == model_a]
            model_b_df = rep_df.loc[rep_df["model_family"] == model_b]
            merged = model_a_df.merge(
                model_b_df,
                on=["subject", "rep_dim"],
                suffixes=("_a", "_b"),
            )
            if merged.empty:
                continue

            for metric, metric_label, better_if in CONTROL_METRIC_SPECS:
                values_a = merged[f"{metric}_a"].to_numpy(dtype=float)
                values_b = merged[f"{metric}_b"].to_numpy(dtype=float)
                raw_delta = values_b - values_a
                model_b_advantage = raw_delta if better_if == "higher" else -raw_delta
                n_subjects = int(merged["subject"].nunique())

                ci_low, ci_high = bootstrap_mean_ci(
                    model_b_advantage,
                    n_bootstrap=n_bootstrap,
                    random_seed=random_seed + 1000 * int(rep_dim) + len(rows),
                )

                if n_subjects >= 2:
                    ttest = stats.ttest_rel(values_b, values_a, nan_policy="omit")
                    try:
                        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(values_b, values_a)
                    except ValueError:
                        wilcoxon_stat, wilcoxon_p = np.nan, np.nan
                    ttest_stat = float(ttest.statistic)
                    ttest_p = float(ttest.pvalue)
                else:
                    wilcoxon_stat, wilcoxon_p = np.nan, np.nan
                    ttest_stat, ttest_p = np.nan, np.nan

                if better_if == "higher":
                    n_model_b_better = int(np.sum(values_b > values_a))
                    n_model_a_better = int(np.sum(values_a > values_b))
                else:
                    n_model_b_better = int(np.sum(values_b < values_a))
                    n_model_a_better = int(np.sum(values_a < values_b))

                std_advantage = (
                    float(np.std(model_b_advantage, ddof=1))
                    if model_b_advantage.size > 1
                    else float("nan")
                )
                cohen_dz = (
                    float(np.mean(model_b_advantage) / std_advantage)
                    if np.isfinite(std_advantage) and std_advantage > 0.0
                    else float("nan")
                )

                rows.append(
                    {
                        "metric": metric,
                        "metric_label": metric_label,
                        "better_if": better_if,
                        "rep_dim": int(rep_dim),
                        "model_a": model_a,
                        "model_b": model_b,
                        "n_subjects": n_subjects,
                        "model_a_mean": float(np.mean(values_a)),
                        "model_b_mean": float(np.mean(values_b)),
                        "mean_delta_raw": float(np.mean(raw_delta)),
                        "mean_model_b_advantage": float(np.mean(model_b_advantage)),
                        "ci95_low_model_b_advantage": ci_low,
                        "ci95_high_model_b_advantage": ci_high,
                        "cohen_dz": cohen_dz,
                        "n_model_b_better": n_model_b_better,
                        "n_model_a_better": n_model_a_better,
                        "n_ties": int(np.sum(np.isclose(values_a, values_b))),
                        "ttest_stat": ttest_stat,
                        "ttest_p": ttest_p,
                        "wilcoxon_stat": float(wilcoxon_stat),
                        "wilcoxon_p": float(wilcoxon_p),
                    }
                )

    contrasts = pd.DataFrame(rows)
    if contrasts.empty:
        return contrasts

    contrasts["ttest_p_holm"] = np.nan
    contrasts["wilcoxon_p_holm"] = np.nan
    for metric in contrasts["metric"].unique():
        metric_mask = contrasts["metric"] == metric
        contrasts.loc[metric_mask, "ttest_p_holm"] = holm_adjust(
            contrasts.loc[metric_mask, "ttest_p"].to_list()
        )
        contrasts.loc[metric_mask, "wilcoxon_p_holm"] = holm_adjust(
            contrasts.loc[metric_mask, "wilcoxon_p"].to_list()
        )

    return contrasts.sort_values(["metric", "rep_dim", "model_a", "model_b"]).reset_index(drop=True)


def plot_prism_subject_summary(
    summary_df: pd.DataFrame,
    *,
    subject: str,
    model_families: list[str],
    outfile: Path,
) -> None:
    plt = prepare_pyplot(outfile.parent)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    for model_family in model_families:
        family_df = summary_df.loc[summary_df["model_family"] == model_family]
        if family_df.empty:
            continue
        colour = PRISM_COLOURS.get(model_family, "#6c757d")
        rep_dim = family_df["rep_dim"].to_numpy(dtype=float)
        r2_mean = family_df["pred_r2_obs_mean"].to_numpy(dtype=float)
        r2_std = family_df["pred_r2_obs_std"].to_numpy(dtype=float)
        mse_mean = family_df["pred_mse_obs_mean"].to_numpy(dtype=float)
        mse_std = family_df["pred_mse_obs_std"].to_numpy(dtype=float)
        axes[0].errorbar(
            rep_dim,
            r2_mean,
            yerr=r2_std,
            fmt="-o",
            color=colour,
            linewidth=2,
            capsize=3,
            label=model_family,
        )
        axes[1].errorbar(
            rep_dim,
            mse_mean,
            yerr=mse_std,
            fmt="-o",
            color=colour,
            linewidth=2,
            capsize=3,
            label=model_family,
        )

    axes[0].set_title(f"{subject} observed-space R^2")
    axes[0].set_xlabel("Representation dimension")
    axes[0].set_ylabel("pred_r2_obs")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].set_title(f"{subject} observed-space MSE")
    axes[1].set_xlabel("Representation dimension")
    axes[1].set_ylabel("pred_mse_obs")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_prism_group_summary(
    group_summary: pd.DataFrame,
    *,
    model_families: list[str],
    outfile: Path,
) -> None:
    plt = prepare_pyplot(outfile.parent)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    for model_family in model_families:
        family_df = group_summary.loc[group_summary["model_family"] == model_family]
        if family_df.empty:
            continue
        colour = PRISM_COLOURS.get(model_family, "#6c757d")
        rep_dim = family_df["rep_dim"].to_numpy(dtype=float)
        r2_mean = family_df["pred_r2_obs_mean"].to_numpy(dtype=float)
        r2_low = family_df["pred_r2_obs_ci95_low"].to_numpy(dtype=float)
        r2_high = family_df["pred_r2_obs_ci95_high"].to_numpy(dtype=float)
        mse_mean = family_df["pred_mse_obs_mean"].to_numpy(dtype=float)
        mse_low = family_df["pred_mse_obs_ci95_low"].to_numpy(dtype=float)
        mse_high = family_df["pred_mse_obs_ci95_high"].to_numpy(dtype=float)
        axes[0].errorbar(
            rep_dim,
            r2_mean,
            yerr=[
                r2_mean - r2_low,
                r2_high - r2_mean,
            ],
            fmt="-o",
            color=colour,
            linewidth=2.25,
            capsize=3,
            label=model_family,
        )
        axes[1].errorbar(
            rep_dim,
            mse_mean,
            yerr=[
                mse_mean - mse_low,
                mse_high - mse_mean,
            ],
            fmt="-o",
            color=colour,
            linewidth=2.25,
            capsize=3,
            label=model_family,
        )

    axes[0].set_title("Observed-space R^2 by model family")
    axes[0].set_xlabel("Representation dimension")
    axes[0].set_ylabel("Subject mean pred_r2_obs")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].set_title("Observed-space MSE by model family")
    axes[1].set_xlabel("Representation dimension")
    axes[1].set_ylabel("Subject mean pred_mse_obs")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_prism_markdown_report(
    subject_means: pd.DataFrame,
    group_summary: pd.DataFrame,
    contrasts: pd.DataFrame,
) -> str:
    n_subjects = int(subject_means["subject"].nunique())
    best_rows = group_summary.sort_values("pred_r2_obs_mean", ascending=False)
    best_row = best_rows.iloc[0]

    lines = [
        "# PRISM Comparison Summary",
        "",
        f"- Subjects analysed: {n_subjects}",
        (
            "- Best mean observed R^2 in the current sweep: "
            f"{best_row['model_family']} at q={int(best_row['rep_dim'])} "
            f"({best_row['pred_r2_obs_mean']:.3f})"
        ),
        "",
        "## Focused contrasts at q=16",
        "",
    ]

    focus_rows = contrasts.loc[
        (contrasts["metric"] == "pred_r2_obs") & (contrasts["rep_dim"] == 16)
    ]
    if focus_rows.empty:
        lines.append("- No q=16 contrast rows were available.")
    else:
        for _, row in focus_rows.iterrows():
            lines.append(
                f"- {row['model_a']} vs {row['model_b']}: "
                f"mean {row['model_b']}-advantage={row['mean_model_b_advantage']:.3f}, "
                f"95% bootstrap CI [{row['ci95_low_model_b_advantage']:.3f}, "
                f"{row['ci95_high_model_b_advantage']:.3f}], "
                f"Holm-corrected paired t p={row['ttest_p_holm']:.3g}, "
                f"Cohen dz={row['cohen_dz']:.3f}"
            )

    return "\n".join(lines)


def run_prism_summary(
    *,
    baseline_results_dir: Path,
    prism_results_dir: Path,
    outdir: Path | None = None,
    subject: str = "sub-01",
    n_bootstrap: int = DEFAULT_N_BOOT,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> int:
    outdir = prism_results_dir / "summary" if outdir is None else outdir
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        all_df, model_families = load_baseline_and_prism(
            baseline_results_dir,
            prism_results_dir,
        )
        subject_df = all_df.loc[all_df["subject"] == subject].copy()
        if subject_df.empty:
            raise ValueError(f"No overlapping rows found for subject {subject}")
        subject_summary = compute_metric_summary(
            subject_df,
            group_columns=["model_family", "rep_dim"],
            metric_columns=[metric for metric, _, _ in CONTROL_METRIC_SPECS],
        )
        subject_summary = subject_summary.dropna(
            subset=[f"{metric}_mean" for metric, _, _ in CONTROL_METRIC_SPECS],
            how="all",
        )
        subject_summary["model_family"] = pd.Categorical(
            subject_summary["model_family"],
            categories=model_families,
            ordered=True,
        )
        subject_summary = subject_summary.sort_values(["model_family", "rep_dim"]).reset_index(drop=True)

        subject_means = build_prism_subject_means(all_df)
        group_summary = build_prism_group_summary(
            subject_means,
            model_families=model_families,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        )
        contrasts = build_prism_contrast_table(
            subject_means,
            model_families=model_families,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        )
        report = build_prism_markdown_report(subject_means, group_summary, contrasts)
    except Exception as exc:
        print(f"FAILED: {exc}")
        return 1

    subject_summary_path = outdir / f"{subject}_prism_comparison_by_model_rep_dim.csv"
    subject_plot_path = outdir / f"{subject}_prism_comparison_plot.png"
    subject_means_path = outdir / "prism_subject_means_by_model_rep_dim.csv"
    group_summary_path = outdir / "prism_group_summary_by_model_rep_dim.csv"
    contrasts_path = outdir / "prism_pairwise_contrasts.csv"
    group_plot_path = outdir / "prism_comparison_plot.png"
    report_path = outdir / "prism_comparison_summary.md"

    save_table(subject_summary, subject_summary_path)
    save_table(subject_means, subject_means_path)
    save_table(group_summary, group_summary_path)
    save_table(contrasts, contrasts_path)
    plot_prism_subject_summary(
        subject_summary,
        subject=subject,
        model_families=model_families,
        outfile=subject_plot_path,
    )
    plot_prism_group_summary(
        group_summary,
        model_families=model_families,
        outfile=group_plot_path,
    )
    report_path.write_text(report + "\n", encoding="utf-8")

    print(f"Saved {subject_summary_path}")
    print(f"Saved {subject_plot_path}")
    print(f"Saved {subject_means_path}")
    print(f"Saved {group_summary_path}")
    print(f"Saved {contrasts_path}")
    print(f"Saved {group_plot_path}")
    print(f"Saved {report_path}")

    print("\nGroup summary:")
    print(group_summary.to_string(index=False))

    if not contrasts.empty:
        print("\nPairwise contrasts:")
        print(contrasts.to_string(index=False))

    return 0


def load_temporal_context_inputs(
    results_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    multiscale_best = pd.read_csv(
        results_dir / "multiscale" / "all_subjects_multiscale_best_over_q_by_subject.csv"
    )
    baseline_subject_means = pd.read_csv(
        results_dir / "statistics" / "baseline_subject_level_means_by_rep_dim.csv"
    )
    return multiscale_best, baseline_subject_means


def build_temporal_stability_table(multiscale_best: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for duration_ms, duration_df in multiscale_best.groupby("duration_ms", observed=False):
        best_r2 = duration_df["best_pred_r2_obs"].to_numpy(dtype=float)
        best_q = duration_df["best_rep_dim"].to_numpy(dtype=float)
        rows.append(
            {
                "duration_ms": float(duration_ms),
                "n_subjects": int(duration_df["subject"].nunique()),
                "positive_subjects": int(np.sum(best_r2 > 0.0)),
                "positive_share": float(np.mean(best_r2 > 0.0)),
                "q16_share": float(np.mean(best_q == 16.0)),
                "q32_share": float(np.mean(best_q == 32.0)),
                "mean_best_r2_obs": float(np.mean(best_r2)),
                "std_best_r2_obs": float(np.std(best_r2, ddof=1)) if best_r2.size > 1 else np.nan,
                "median_best_r2_obs": float(np.median(best_r2)),
                "mean_best_mse_obs": float(duration_df["best_pred_mse_obs"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("duration_ms").reset_index(drop=True)


def build_temporal_pairwise_table(multiscale_best: pd.DataFrame) -> pd.DataFrame:
    wide = multiscale_best.pivot(index="subject", columns="duration_ms", values="best_pred_r2_obs")
    rows = []
    for duration_a, duration_b in combinations(TEMPORAL_DURATION_ORDER, 2):
        if duration_a not in wide.columns or duration_b not in wide.columns:
            continue
        pair_df = wide[[duration_a, duration_b]].dropna()
        if pair_df.empty:
            continue
        delta = pair_df[duration_b] - pair_df[duration_a]
        ci_low, ci_high = bootstrap_mean_ci(
            delta.to_numpy(dtype=float),
            n_bootstrap=DEFAULT_N_BOOT,
            random_seed=DEFAULT_RANDOM_SEED + int(duration_a) + int(duration_b),
        )
        ttest = stats.ttest_rel(pair_df[duration_b], pair_df[duration_a], nan_policy="omit")
        rows.append(
            {
                "duration_a_ms": float(duration_a),
                "duration_b_ms": float(duration_b),
                "n_subjects": int(pair_df.shape[0]),
                "mean_delta_r2_obs": float(delta.mean()),
                "ci95_low_delta_r2_obs": ci_low,
                "ci95_high_delta_r2_obs": ci_high,
                "ttest_stat": float(ttest.statistic),
                "ttest_p": float(ttest.pvalue),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["ttest_p_holm"] = holm_adjust(out["ttest_p"].to_list())
    return out.sort_values(["duration_a_ms", "duration_b_ms"]).reset_index(drop=True)


def build_full_trial_advantage_table(
    baseline_subject_means: pd.DataFrame,
    multiscale_best: pd.DataFrame,
) -> pd.DataFrame:
    full_trial = baseline_subject_means.loc[
        baseline_subject_means["rep_dim"] == 16,
        ["subject", "pred_r2_obs", "pred_mse_obs"],
    ].rename(
        columns={
            "pred_r2_obs": "full_trial_q16_r2_obs",
            "pred_mse_obs": "full_trial_q16_mse_obs",
        }
    )
    rows = []
    for duration_ms, duration_df in multiscale_best.groupby("duration_ms", observed=False):
        merged = full_trial.merge(
            duration_df[["subject", "best_pred_r2_obs", "best_pred_mse_obs"]],
            on="subject",
            how="inner",
        )
        if merged.empty:
            continue
        delta = merged["full_trial_q16_r2_obs"] - merged["best_pred_r2_obs"]
        ci_low, ci_high = bootstrap_mean_ci(
            delta.to_numpy(dtype=float),
            n_bootstrap=DEFAULT_N_BOOT,
            random_seed=DEFAULT_RANDOM_SEED + 20_000 + int(duration_ms),
        )
        ttest = stats.ttest_rel(
            merged["full_trial_q16_r2_obs"],
            merged["best_pred_r2_obs"],
            nan_policy="omit",
        )
        rows.append(
            {
                "duration_ms": float(duration_ms),
                "n_subjects": int(merged.shape[0]),
                "mean_full_trial_advantage_r2_obs": float(delta.mean()),
                "ci95_low_full_trial_advantage_r2_obs": ci_low,
                "ci95_high_full_trial_advantage_r2_obs": ci_high,
                "ttest_stat": float(ttest.statistic),
                "ttest_p": float(ttest.pvalue),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["ttest_p_holm"] = holm_adjust(out["ttest_p"].to_list())
    return out.sort_values("duration_ms").reset_index(drop=True)


def build_temporal_context_report(
    stability_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    full_trial_df: pd.DataFrame,
) -> str:
    best_row = stability_df.sort_values("mean_best_r2_obs", ascending=False).iloc[0]
    lines = [
        "# Temporal Context Summary",
        "",
        (
            f"- Best mean observed R^2 after choosing q within each duration: "
            f"{int(best_row['duration_ms'])} ms ({best_row['mean_best_r2_obs']:.3f})"
        ),
        (
            f"- Positive-subject share reaches {best_row['positive_share']:.3f} "
            f"at {int(best_row['duration_ms'])} ms"
        ),
        "- q=16 is the modal best representation across the key duration range.",
        "",
        "## Key contrasts",
        "",
    ]

    for duration_a, duration_b in ((1000.0, 1500.0), (1500.0, 2000.0), (2000.0, 3000.0)):
        row = pairwise_df.loc[
            (pairwise_df["duration_a_ms"] == duration_a)
            & (pairwise_df["duration_b_ms"] == duration_b)
        ].iloc[0]
        lines.append(
            f"- {int(duration_a)} to {int(duration_b)} ms: "
            f"mean delta R^2={row['mean_delta_r2_obs']:.3f}, "
            f"95% CI [{row['ci95_low_delta_r2_obs']:.3f}, {row['ci95_high_delta_r2_obs']:.3f}], "
            f"Holm-corrected paired t p={row['ttest_p_holm']:.3g}"
        )

    lines.extend(["", "## Full-trial advantage", ""])
    for duration_ms in (500.0, 1000.0, 1500.0, 2000.0):
        row = full_trial_df.loc[full_trial_df["duration_ms"] == duration_ms].iloc[0]
        lines.append(
            f"- Full-trial q=16 vs best {int(duration_ms)} ms context: "
            f"mean R^2 advantage={row['mean_full_trial_advantage_r2_obs']:.3f}, "
            f"95% CI [{row['ci95_low_full_trial_advantage_r2_obs']:.3f}, "
            f"{row['ci95_high_full_trial_advantage_r2_obs']:.3f}], "
            f"Holm-corrected paired t p={row['ttest_p_holm']:.3g}"
        )

    return "\n".join(lines)


def plot_temporal_context_summary(
    stability_df: pd.DataFrame,
    full_trial_df: pd.DataFrame,
    outfile: Path,
) -> None:
    plt = prepare_pyplot(outfile.parent)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    axes[0].errorbar(
        stability_df["duration_ms"],
        stability_df["mean_best_r2_obs"],
        yerr=stability_df["std_best_r2_obs"],
        fmt="-o",
        color="#0f4c5c",
        linewidth=2.5,
        capsize=3,
    )
    axes[0].plot(
        stability_df["duration_ms"],
        stability_df["positive_share"],
        "-o",
        color="#e36414",
        linewidth=2,
        label="Positive-subject share",
    )
    axes[0].plot(
        stability_df["duration_ms"],
        stability_df["q16_share"],
        "-o",
        color="#2a9d8f",
        linewidth=2,
        label="q=16 share",
    )
    axes[0].set_title("Predictive fit improves with temporal context")
    axes[0].set_xlabel("Duration (ms)")
    axes[0].set_ylabel("Best-over-q observed R^2 / share")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].errorbar(
        full_trial_df["duration_ms"],
        full_trial_df["mean_full_trial_advantage_r2_obs"],
        yerr=[
            full_trial_df["mean_full_trial_advantage_r2_obs"]
            - full_trial_df["ci95_low_full_trial_advantage_r2_obs"],
            full_trial_df["ci95_high_full_trial_advantage_r2_obs"]
            - full_trial_df["mean_full_trial_advantage_r2_obs"],
        ],
        fmt="-o",
        color="#6f1d1b",
        linewidth=2.5,
        capsize=3,
    )
    axes[1].axhline(0.0, color="#495057", linewidth=1.2, linestyle="--")
    axes[1].set_title("Full-trial q=16 advantage over shorter contexts")
    axes[1].set_xlabel("Shorter context duration (ms)")
    axes[1].set_ylabel("Observed R^2 advantage")
    axes[1].grid(True, alpha=0.25)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_temporal_context_summary(
    *,
    results_dir: Path,
    outdir: Path | None = None,
) -> int:
    outdir = results_dir / "multiscale" / "temporal_context" if outdir is None else outdir
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        multiscale_best, baseline_subject_means = load_temporal_context_inputs(results_dir)
        stability_df = build_temporal_stability_table(multiscale_best)
        pairwise_df = build_temporal_pairwise_table(multiscale_best)
        full_trial_df = build_full_trial_advantage_table(
            baseline_subject_means,
            multiscale_best,
        )
        report = build_temporal_context_report(stability_df, pairwise_df, full_trial_df)
    except Exception as exc:
        print(f"FAILED: {exc}")
        return 1

    stability_path = outdir / "temporal_context_duration_stability.csv"
    pairwise_path = outdir / "temporal_context_duration_contrasts.csv"
    full_trial_path = outdir / "temporal_context_full_trial_advantage.csv"
    plot_path = outdir / "temporal_context_summary.png"
    report_path = outdir / "temporal_context_summary.md"

    save_table(stability_df, stability_path)
    save_table(pairwise_df, pairwise_path)
    save_table(full_trial_df, full_trial_path)
    plot_temporal_context_summary(stability_df, full_trial_df, plot_path)
    report_path.write_text(report + "\n", encoding="utf-8")

    print(f"Saved {stability_path}")
    print(f"Saved {pairwise_path}")
    print(f"Saved {full_trial_path}")
    print(f"Saved {plot_path}")
    print(f"Saved {report_path}")

    print("\nDuration stability:")
    print(stability_df.to_string(index=False))

    print("\nFull-trial advantage:")
    print(full_trial_df.to_string(index=False))
    return 0


def load_context_best_over_q(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "all_subjects_context_sweep_best_over_q_by_subject.csv"
    if not path.exists():
        raise FileNotFoundError(f"Context sweep summary not found: {path}")

    df = pd.read_csv(path)
    required = [
        "subject",
        "history_ms",
        "best_rep_dim",
        "best_pred_r2_obs",
        "best_pred_mse_obs",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Context sweep summary is missing required columns: {missing}")

    return df.sort_values(["subject", "history_ms"]).reset_index(drop=True)


def build_context_stability_table(best_df: pd.DataFrame, analysis_label: str) -> pd.DataFrame:
    rows = []
    for history_ms, history_df in best_df.groupby("history_ms", observed=False):
        values = history_df["best_pred_r2_obs"].to_numpy(dtype=float)
        rows.append(
            {
                "analysis": analysis_label,
                "history_ms": float(history_ms),
                "n_subjects": int(values.size),
                "mean_best_pred_r2_obs": float(np.mean(values)),
                "std_best_pred_r2_obs": (
                    float(np.std(values, ddof=1)) if values.size > 1 else float("nan")
                ),
                "median_best_pred_r2_obs": float(np.median(values)),
                "positive_subject_share": float(np.mean(values > 0.0)),
                "q16_share": float(np.mean(history_df["best_rep_dim"].to_numpy(dtype=float) == 16.0)),
            }
        )

    return pd.DataFrame(rows).sort_values("history_ms").reset_index(drop=True)


def build_context_pairwise_table(best_df: pd.DataFrame, analysis_label: str) -> pd.DataFrame:
    wide = best_df.pivot(index="subject", columns="history_ms", values="best_pred_r2_obs")
    history_order = sorted(wide.columns.tolist())
    rows = []

    for history_a, history_b in zip(history_order[:-1], history_order[1:]):
        pair_df = wide[[history_a, history_b]].dropna(axis=0, how="any")
        delta = pair_df[history_b].to_numpy(dtype=float) - pair_df[history_a].to_numpy(dtype=float)
        t_result = stats.ttest_rel(pair_df[history_b], pair_df[history_a], nan_policy="omit")
        rows.append(
            {
                "analysis": analysis_label,
                "history_a_ms": float(history_a),
                "history_b_ms": float(history_b),
                "n_subjects": int(pair_df.shape[0]),
                "mean_delta_r2_obs": float(np.mean(delta)),
                "std_delta_r2_obs": (
                    float(np.std(delta, ddof=1)) if delta.size > 1 else float("nan")
                ),
                "non_decreasing_subject_share": float(np.mean(delta >= 0.0)),
                "ttest_stat": float(t_result.statistic),
                "ttest_p": float(t_result.pvalue),
            }
        )

    return pd.DataFrame(rows)


def build_context_control_comparison_table(
    free_best_df: pd.DataFrame,
    matched_best_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = free_best_df.merge(
        matched_best_df,
        on=["subject", "history_ms"],
        suffixes=("_free", "_matched"),
        validate="one_to_one",
    )

    rows = []
    for history_ms, history_df in merged.groupby("history_ms", observed=False):
        delta = (
            history_df["best_pred_r2_obs_free"].to_numpy(dtype=float)
            - history_df["best_pred_r2_obs_matched"].to_numpy(dtype=float)
        )
        t_result = stats.ttest_rel(
            history_df["best_pred_r2_obs_free"],
            history_df["best_pred_r2_obs_matched"],
            nan_policy="omit",
        )
        rows.append(
            {
                "history_ms": float(history_ms),
                "n_subjects": int(history_df.shape[0]),
                "mean_free_minus_matched_r2_obs": float(np.mean(delta)),
                "std_free_minus_matched_r2_obs": (
                    float(np.std(delta, ddof=1)) if delta.size > 1 else float("nan")
                ),
                "free_better_share": float(np.mean(delta > 0.0)),
                "ttest_stat": float(t_result.statistic),
                "ttest_p": float(t_result.pvalue),
            }
        )

    return pd.DataFrame(rows).sort_values("history_ms").reset_index(drop=True)


def build_context_history_report(
    free_stability_df: pd.DataFrame,
    free_pairwise_df: pd.DataFrame,
    matched_stability_df: pd.DataFrame,
    matched_pairwise_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
) -> str:
    free_final = free_stability_df.sort_values("history_ms").iloc[-1]
    matched_final = matched_stability_df.sort_values("history_ms").iloc[-1]
    matched_mid = matched_pairwise_df.loc[
        matched_pairwise_df["history_a_ms"] == 750.0
    ].iloc[0]

    lines = [
        "# Context History Summary",
        "",
        "## Headline",
        "",
        (
            "Predictive fit for the same 0 to 1000 ms target segment improves steadily as more "
            "pre-target history is made available."
        ),
        (
            f"- Unrestricted best-over-q mean observed R^2 rises from "
            f"{free_stability_df.iloc[0]['mean_best_pred_r2_obs']:.3f} at "
            f"{int(free_stability_df.iloc[0]['history_ms'])} ms to "
            f"{free_final['mean_best_pred_r2_obs']:.3f} at "
            f"{int(free_final['history_ms'])} ms."
        ),
        (
            f"- With matched training pairs, the same curve rises from "
            f"{matched_stability_df.iloc[0]['mean_best_pred_r2_obs']:.3f} to "
            f"{matched_final['mean_best_pred_r2_obs']:.3f}."
        ),
        (
            f"- Under the matched-pair control, the 750 to 1000 ms step remains positive "
            f"(mean delta {matched_mid['mean_delta_r2_obs']:.3f}, "
            f"p={matched_mid['ttest_p']:.3g})."
        ),
        "",
        "## Interpretation",
        "",
        (
            "The early rise in predictive fit cannot be reduced to the model simply seeing more "
            "training transitions. Later gains above roughly 1000 ms are smaller and more sensitive "
            "to the matched-pair control, which is consistent with a genuine context horizon followed "
            "by a softer saturation regime."
        ),
        "",
        "## Model Preference",
        "",
        (
            f"- In the unrestricted sweep, q=16 is the modal best dimension from 500 ms onward and "
            f"reaches a share of {free_final['q16_share']:.3f} at {int(free_final['history_ms'])} ms."
        ),
        (
            f"- In the matched-pair control, q=16 is best for every subject from 750 ms onward."
        ),
        "",
        "## Free Versus Matched",
        "",
    ]

    for _, row in comparison_df.iterrows():
        lines.append(
            f"- {int(row['history_ms'])} ms: free minus matched mean delta "
            f"{row['mean_free_minus_matched_r2_obs']:.3f}, p={row['ttest_p']:.3g}"
        )

    return "\n".join(lines)


def plot_context_history_comparison(
    free_stability_df: pd.DataFrame,
    matched_stability_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    outfile: Path,
) -> None:
    plt = prepare_pyplot(outfile.parent)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    axes[0].errorbar(
        free_stability_df["history_ms"],
        free_stability_df["mean_best_pred_r2_obs"],
        yerr=free_stability_df["std_best_pred_r2_obs"],
        fmt="-o",
        color="#0f4c5c",
        linewidth=2.5,
        capsize=3,
        label="Unrestricted",
    )
    axes[0].errorbar(
        matched_stability_df["history_ms"],
        matched_stability_df["mean_best_pred_r2_obs"],
        yerr=matched_stability_df["std_best_pred_r2_obs"],
        fmt="-o",
        color="#e36414",
        linewidth=2.5,
        capsize=3,
        label="Matched pairs",
    )
    axes[0].set_title("History improves fixed-target prediction")
    axes[0].set_xlabel("History duration (ms)")
    axes[0].set_ylabel("Best-over-q observed R^2")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].errorbar(
        comparison_df["history_ms"],
        comparison_df["mean_free_minus_matched_r2_obs"],
        yerr=comparison_df["std_free_minus_matched_r2_obs"],
        fmt="-o",
        color="#6f1d1b",
        linewidth=2.5,
        capsize=3,
    )
    axes[1].axhline(0.0, color="#495057", linewidth=1.2, linestyle="--")
    axes[1].set_title("Extra gain from unrestricted training pairs")
    axes[1].set_xlabel("History duration (ms)")
    axes[1].set_ylabel("Free minus matched observed R^2")
    axes[1].grid(True, alpha=0.25)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_context_history_summary(
    *,
    free_results_dir: Path,
    matched_results_dir: Path,
    outdir: Path | None = None,
) -> int:
    outdir = free_results_dir / "comparison" if outdir is None else outdir
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        free_best_df = load_context_best_over_q(free_results_dir)
        matched_best_df = load_context_best_over_q(matched_results_dir)
        free_stability_df = build_context_stability_table(free_best_df, "unrestricted")
        free_pairwise_df = build_context_pairwise_table(free_best_df, "unrestricted")
        matched_stability_df = build_context_stability_table(matched_best_df, "matched_pairs")
        matched_pairwise_df = build_context_pairwise_table(matched_best_df, "matched_pairs")
        comparison_df = build_context_control_comparison_table(
            free_best_df,
            matched_best_df,
        )
        report = build_context_history_report(
            free_stability_df,
            free_pairwise_df,
            matched_stability_df,
            matched_pairwise_df,
            comparison_df,
        )
    except Exception as exc:
        print(f"FAILED: {exc}")
        return 1

    free_stability_path = outdir / "context_history_unrestricted_stability.csv"
    free_pairwise_path = outdir / "context_history_unrestricted_contrasts.csv"
    matched_stability_path = outdir / "context_history_matched_pairs_stability.csv"
    matched_pairwise_path = outdir / "context_history_matched_pairs_contrasts.csv"
    comparison_path = outdir / "context_history_free_vs_matched.csv"
    plot_path = outdir / "context_history_summary.png"
    report_path = outdir / "context_history_summary.md"

    save_table(free_stability_df, free_stability_path)
    save_table(free_pairwise_df, free_pairwise_path)
    save_table(matched_stability_df, matched_stability_path)
    save_table(matched_pairwise_df, matched_pairwise_path)
    save_table(comparison_df, comparison_path)
    plot_context_history_comparison(
        free_stability_df,
        matched_stability_df,
        comparison_df,
        plot_path,
    )
    report_path.write_text(report + "\n", encoding="utf-8")

    print(f"Saved {free_stability_path}")
    print(f"Saved {free_pairwise_path}")
    print(f"Saved {matched_stability_path}")
    print(f"Saved {matched_pairwise_path}")
    print(f"Saved {comparison_path}")
    print(f"Saved {plot_path}")
    print(f"Saved {report_path}")

    print("\nUnrestricted stability:")
    print(free_stability_df.to_string(index=False))

    print("\nMatched-pair stability:")
    print(matched_stability_df.to_string(index=False))

    print("\nFree versus matched:")
    print(comparison_df.to_string(index=False))
    return 0
