from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_ROOT = Path(
    "data/results_prism/eeg_prism_region_timecourse_pca_q4/summary_subject_decoder_region_timecourse"
)
DEFAULT_OUTDIR = DEFAULT_ROOT / "fixed_cell_occipital_125_375"
DEFAULT_REGION = "occipital"
DEFAULT_WINDOW = "125-375"
DEFAULT_PERMUTATIONS = 20000
DEFAULT_MAX_STAT_PERMUTATIONS = 5000

MODEL_LABELS = {
    "raw_central_delta": "Raw EEG",
    "var_central_augmented": "VAR summaries",
    "prism_central_augmented": "PRISM summaries",
    "raw_plus_var": "Raw EEG + VAR",
    "raw_plus_prism": "Raw EEG + PRISM",
}


def _ci95(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return math.nan, math.nan
    half_width = stats.t.ppf(0.975, values.size - 1) * stats.sem(values)
    return float(values.mean() - half_width), float(values.mean() + half_width)


def _format_p(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    if value < 1e-3:
        return f"{value:.2e}"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    labels = labels[finite]
    scores = scores[finite]
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = stats.rankdata(scores)
    pos_ranks = float(np.sum(ranks[labels == 1]))
    return float((pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _residualise(values: np.ndarray, covariate: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    covariate = np.asarray(covariate, dtype=float)
    finite = np.isfinite(values) & np.isfinite(covariate)
    residuals = np.full(values.shape, np.nan, dtype=float)
    if finite.sum() < 3:
        return residuals
    x = np.column_stack([np.ones(finite.sum(), dtype=float), covariate[finite]])
    beta, *_ = np.linalg.lstsq(x, values[finite], rcond=None)
    residuals[finite] = values[finite] - x @ beta
    return residuals


def _mean_hit_minus_miss(labels: np.ndarray, values: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    labels = labels[finite]
    values = values[finite]
    hit = values[labels == 1]
    miss = values[labels == 0]
    if hit.size == 0 or miss.size == 0:
        return math.nan
    return float(hit.mean() - miss.mean())


def _stimamp_matched_delta(labels: np.ndarray, values: np.ndarray, stimamp: np.ndarray) -> tuple[float, float, int]:
    labels = np.asarray(labels, dtype=int)
    values = np.asarray(values, dtype=float)
    stimamp = np.asarray(stimamp, dtype=float)
    finite = np.isfinite(values) & np.isfinite(stimamp)
    labels = labels[finite]
    values = values[finite]
    stimamp = stimamp[finite]

    hit_idx = np.flatnonzero(labels == 1)
    miss_idx = np.flatnonzero(labels == 0)
    if hit_idx.size == 0 or miss_idx.size == 0:
        return math.nan, math.nan, 0

    deltas: list[float] = []
    gaps: list[float] = []
    if hit_idx.size <= miss_idx.size:
        available = miss_idx.tolist()
        for h in hit_idx[np.argsort(stimamp[hit_idx])]:
            nearest_pos = int(np.argmin(np.abs(stimamp[available] - stimamp[h])))
            m = available.pop(nearest_pos)
            deltas.append(float(values[h] - values[m]))
            gaps.append(float(abs(stimamp[h] - stimamp[m])))
    else:
        available = hit_idx.tolist()
        for m in miss_idx[np.argsort(stimamp[miss_idx])]:
            nearest_pos = int(np.argmin(np.abs(stimamp[available] - stimamp[m])))
            h = available.pop(nearest_pos)
            deltas.append(float(values[h] - values[m]))
            gaps.append(float(abs(stimamp[h] - stimamp[m])))
    return float(np.mean(deltas)), float(np.mean(gaps)), int(len(deltas))


def _summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    ci_low, ci_high = _ci95(values)
    ttest = stats.ttest_1samp(values, 0.0) if values.size >= 2 else None
    try:
        wilcoxon_p = float(stats.wilcoxon(values).pvalue)
    except ValueError:
        wilcoxon_p = math.nan
    positive = int(np.sum(values > 0.0))
    sign_p = (
        float(stats.binomtest(positive, values.size, p=0.5, alternative="greater").pvalue)
        if values.size
        else math.nan
    )
    return {
        "n_subjects": int(values.size),
        "mean": float(values.mean()) if values.size else math.nan,
        "median": float(np.median(values)) if values.size else math.nan,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "positive_subjects": positive,
        "ttest_p": float(ttest.pvalue) if ttest is not None else math.nan,
        "wilcoxon_p": wilcoxon_p,
        "sign_p_greater": sign_p,
    }


def _load_scores(root: Path) -> pd.DataFrame:
    path = root / "eeg_prism_region_timecourse_trial_scores.csv"
    df = pd.read_csv(path)
    index_cols = [
        "region_name",
        "window_label",
        "target_center_ms",
        "subject",
        "trial_idx",
        "label",
        "sdt",
        "confidence",
        "stimamp",
    ]
    wide = df.pivot_table(index=index_cols, columns="model_name", values="decoder_score").reset_index()
    required = {"raw_central_delta", "raw_plus_prism", "raw_plus_var", "prism_central_augmented", "var_central_augmented"}
    missing = sorted(required.difference(wide.columns))
    if missing:
        raise ValueError(f"Missing score columns: {missing}")
    return wide


def _load_wide(root: Path, region: str, window_label: str) -> pd.DataFrame:
    wide = _load_scores(root)
    wide = wide.loc[wide["region_name"].eq(region) & wide["window_label"].eq(window_label)].copy()
    if wide.empty:
        raise ValueError(f"No rows matched region={region!r}, window={window_label!r}.")
    return wide


def _subject_auc_table(wide: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    model_names = list(MODEL_LABELS)
    for subject, group in wide.groupby("subject", sort=True):
        labels = group["label"].to_numpy(dtype=int)
        row: dict[str, object] = {"subject": subject, "n_trials": int(group.shape[0])}
        for model in model_names:
            row[f"auc_{model}"] = _roc_auc(labels, group[model].to_numpy(dtype=float))
        rows.append(row)
    return pd.DataFrame(rows)


def _subject_contribution_table(wide: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subject, group in wide.groupby("subject", sort=True):
        labels = group["label"].to_numpy(dtype=int)
        stimamp = group["stimamp"].to_numpy(dtype=float)
        prism = group["raw_plus_prism"].to_numpy(dtype=float) - group["raw_central_delta"].to_numpy(dtype=float)
        var = group["raw_plus_var"].to_numpy(dtype=float) - group["raw_central_delta"].to_numpy(dtype=float)
        prism_matched, prism_gap, prism_pairs = _stimamp_matched_delta(labels, prism, stimamp)
        var_matched, var_gap, var_pairs = _stimamp_matched_delta(labels, var, stimamp)
        rows.append(
            {
                "subject": subject,
                "prism_hit_minus_miss": _mean_hit_minus_miss(labels, prism),
                "var_hit_minus_miss": _mean_hit_minus_miss(labels, var),
                "prism_minus_var_hit_minus_miss": _mean_hit_minus_miss(labels, prism - var),
                "prism_stimamp_resid_hit_minus_miss": _mean_hit_minus_miss(labels, _residualise(prism, stimamp)),
                "var_stimamp_resid_hit_minus_miss": _mean_hit_minus_miss(labels, _residualise(var, stimamp)),
                "prism_stimamp_matched_hit_minus_miss": prism_matched,
                "var_stimamp_matched_hit_minus_miss": var_matched,
                "prism_stimamp_matched_pairs": prism_pairs,
                "prism_stimamp_matched_mean_abs_gap": prism_gap,
                "var_stimamp_matched_pairs": var_pairs,
                "var_stimamp_matched_mean_abs_gap": var_gap,
            }
        )
    return pd.DataFrame(rows)


def _contrast_table(subject_auc: pd.DataFrame, contribution: pd.DataFrame) -> pd.DataFrame:
    contrasts = {
        "auc_prism_gain_over_raw": subject_auc["auc_raw_plus_prism"] - subject_auc["auc_raw_central_delta"],
        "auc_var_gain_over_raw": subject_auc["auc_raw_plus_var"] - subject_auc["auc_raw_central_delta"],
        "auc_prism_minus_var_hybrid": subject_auc["auc_raw_plus_prism"] - subject_auc["auc_raw_plus_var"],
        "auc_prism_summary_minus_var_summary": subject_auc["auc_prism_central_augmented"]
        - subject_auc["auc_var_central_augmented"],
        "prism_hit_minus_miss": contribution["prism_hit_minus_miss"],
        "var_hit_minus_miss": contribution["var_hit_minus_miss"],
        "prism_minus_var_hit_minus_miss": contribution["prism_minus_var_hit_minus_miss"],
        "prism_stimamp_resid_hit_minus_miss": contribution["prism_stimamp_resid_hit_minus_miss"],
        "var_stimamp_resid_hit_minus_miss": contribution["var_stimamp_resid_hit_minus_miss"],
        "prism_stimamp_matched_hit_minus_miss": contribution["prism_stimamp_matched_hit_minus_miss"],
        "var_stimamp_matched_hit_minus_miss": contribution["var_stimamp_matched_hit_minus_miss"],
    }
    rows: list[dict[str, object]] = []
    for name, values in contrasts.items():
        row = {"contrast": name}
        row.update(_summary(values.to_numpy(dtype=float)))
        rows.append(row)
    return pd.DataFrame(rows)


def _auc_delta_for_indices(rank_delta: np.ndarray, selected: np.ndarray, n_pos: int, n_neg: int) -> float:
    return float(rank_delta[selected].sum() / (n_pos * n_neg))


def _mean_delta_for_indices(values: np.ndarray, selected: np.ndarray, n_pos: int, n_neg: int) -> float:
    selected_mask = np.zeros(values.shape[0], dtype=bool)
    selected_mask[selected] = True
    return float(values[selected_mask].mean() - values[~selected_mask].mean())


def _permutation_summary(
    wide: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    nulls = {
        "auc_prism_gain_over_raw": np.zeros(n_permutations, dtype=float),
        "auc_var_gain_over_raw": np.zeros(n_permutations, dtype=float),
        "prism_hit_minus_miss": np.zeros(n_permutations, dtype=float),
        "prism_stimamp_resid_hit_minus_miss": np.zeros(n_permutations, dtype=float),
    }
    subjects = list(wide["subject"].drop_duplicates())

    for subject in subjects:
        group = wide.loc[wide["subject"].eq(subject)].reset_index(drop=True)
        labels = group["label"].to_numpy(dtype=int)
        n_pos = int(np.sum(labels == 1))
        n_neg = int(np.sum(labels == 0))
        if n_pos == 0 or n_neg == 0:
            continue

        raw_rank = stats.rankdata(group["raw_central_delta"].to_numpy(dtype=float))
        prism_rank = stats.rankdata(group["raw_plus_prism"].to_numpy(dtype=float))
        var_rank = stats.rankdata(group["raw_plus_var"].to_numpy(dtype=float))
        prism_score = group["raw_plus_prism"].to_numpy(dtype=float) - group["raw_central_delta"].to_numpy(dtype=float)
        stimamp_resid = _residualise(prism_score, group["stimamp"].to_numpy(dtype=float))
        valid_indices = np.flatnonzero(np.isfinite(stimamp_resid))

        for idx in range(n_permutations):
            selected = rng.choice(labels.size, size=n_pos, replace=False)
            nulls["auc_prism_gain_over_raw"][idx] += _auc_delta_for_indices(prism_rank - raw_rank, selected, n_pos, n_neg)
            nulls["auc_var_gain_over_raw"][idx] += _auc_delta_for_indices(var_rank - raw_rank, selected, n_pos, n_neg)
            nulls["prism_hit_minus_miss"][idx] += _mean_delta_for_indices(prism_score, selected, n_pos, n_neg)
            if valid_indices.size == labels.size:
                nulls["prism_stimamp_resid_hit_minus_miss"][idx] += _mean_delta_for_indices(
                    stimamp_resid,
                    selected,
                    n_pos,
                    n_neg,
                )

    n_subjects = len(subjects)
    rows: list[dict[str, object]] = []
    observed_by_name = observed.set_index("contrast")["mean"].to_dict()
    for name, values in nulls.items():
        values = values / n_subjects
        obs = float(observed_by_name[name])
        rows.append(
            {
                "contrast": name,
                "observed_mean": obs,
                "null_mean": float(values.mean()),
                "null_sd": float(values.std(ddof=1)),
                "perm_p_greater": float((1 + np.sum(values >= obs)) / (n_permutations + 1)),
                "perm_p_two_sided": float((1 + np.sum(np.abs(values) >= abs(obs))) / (n_permutations + 1)),
                "n_permutations": int(n_permutations),
            }
        )
    return pd.DataFrame(rows)


def _max_stat_search_control(
    wide_all: pd.DataFrame,
    *,
    target_region: str,
    target_window: str,
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    bundles: list[dict[str, object]] = []
    observed_subject: dict[tuple[str, str, float], list[float]] = {}

    for (region, window, centre, subject), group in wide_all.groupby(
        ["region_name", "window_label", "target_center_ms", "subject"],
        sort=True,
    ):
        labels = group["label"].to_numpy(dtype=int)
        values = group["raw_plus_prism"].to_numpy(dtype=float) - group["raw_central_delta"].to_numpy(dtype=float)
        finite = np.isfinite(values)
        labels = labels[finite]
        values = values[finite]
        n_pos = int(np.sum(labels == 1))
        n_neg = int(np.sum(labels == 0))
        if n_pos == 0 or n_neg == 0:
            continue
        key = (str(region), str(window), float(centre))
        observed_subject.setdefault(key, []).append(_mean_hit_minus_miss(labels, values))
        bundles.append(
            {
                "key": key,
                "values": values,
                "total": float(values.sum()),
                "n_pos": n_pos,
                "n_neg": n_neg,
                "n": int(values.size),
            }
        )

    observed_rows: list[dict[str, object]] = []
    for key, values in observed_subject.items():
        arr = np.asarray(values, dtype=float)
        t_stat = float(stats.ttest_1samp(arr, 0.0).statistic) if arr.size >= 2 else math.nan
        observed_rows.append(
            {
                "region_name": key[0],
                "window_label": key[1],
                "target_center_ms": key[2],
                "observed_mean": float(arr.mean()),
                "observed_t": t_stat,
                "n_subjects": int(arr.size),
            }
        )
    observed = pd.DataFrame(observed_rows).sort_values("observed_t", ascending=False)
    target = observed.loc[
        observed["region_name"].eq(target_region) & observed["window_label"].eq(target_window)
    ]
    if target.empty:
        raise ValueError("Target cell not present in search-control table.")
    target_row = target.iloc[0]
    target_t = float(target_row["observed_t"])
    target_rank = int(observed.index.get_loc(target.index[0]) + 1)

    rng = np.random.default_rng(seed)
    cell_keys = list(observed_subject)
    key_to_pos = {key: idx for idx, key in enumerate(cell_keys)}
    null_max = np.zeros(n_permutations, dtype=float)
    for perm_idx in range(n_permutations):
        per_cell: list[list[float]] = [[] for _ in cell_keys]
        for bundle in bundles:
            n = int(bundle["n"])
            n_pos = int(bundle["n_pos"])
            n_neg = int(bundle["n_neg"])
            values = bundle["values"]
            selected = rng.choice(n, size=n_pos, replace=False)
            selected_sum = float(values[selected].sum())
            delta = selected_sum / n_pos - (float(bundle["total"]) - selected_sum) / n_neg
            per_cell[key_to_pos[bundle["key"]]].append(delta)
        t_values = []
        for values in per_cell:
            arr = np.asarray(values, dtype=float)
            if arr.size < 2 or np.std(arr, ddof=1) == 0.0:
                continue
            t_values.append(float(arr.mean() / (arr.std(ddof=1) / math.sqrt(arr.size))))
        null_max[perm_idx] = max(t_values) if t_values else math.nan

    return pd.DataFrame(
        [
            {
                "target_region": target_region,
                "target_window": target_window,
                "target_observed_mean": float(target_row["observed_mean"]),
                "target_observed_t": target_t,
                "target_rank_by_observed_t": target_rank,
                "n_cells": int(observed.shape[0]),
                "max_stat_p_greater": float((1 + np.sum(null_max >= target_t)) / (n_permutations + 1)),
                "null_max_t_mean": float(np.nanmean(null_max)),
                "null_max_t_95": float(np.nanquantile(null_max, 0.95)),
                "n_permutations": int(n_permutations),
            }
        ]
    )


def _plot_subjects(subject_auc: pd.DataFrame, contribution: pd.DataFrame, outdir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    auc = subject_auc.copy()
    auc["gain"] = auc["auc_raw_plus_prism"] - auc["auc_raw_central_delta"]
    auc = auc.sort_values("gain")

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), constrained_layout=True)
    x = np.arange(auc.shape[0])
    axes[0].axhline(0.0, color="0.35", linewidth=0.8)
    axes[0].bar(x, auc["gain"], color="#4C78A8", width=0.72)
    axes[0].set_ylabel("AUC gain over raw")
    axes[0].set_xlabel("Subject")
    axes[0].set_xticks([])
    axes[0].set_title("Raw EEG + PRISM")

    contrib = contribution.set_index("subject").loc[auc["subject"]]
    axes[1].axhline(0.0, color="0.35", linewidth=0.8)
    axes[1].bar(x, contrib["prism_hit_minus_miss"], color="#59A14F", width=0.72)
    axes[1].set_ylabel("Hit-minus-miss contribution")
    axes[1].set_xlabel("Subject")
    axes[1].set_xticks([])
    axes[1].set_title("PRISM contribution")

    for label, axis in zip(("A", "B"), axes):
        axis.text(-0.16, 1.04, label, transform=axis.transAxes, fontsize=11, fontweight="bold")
    fig.savefig(outdir / "eeg_prism_fixed_cell_subjects.png", dpi=300)
    fig.savefig(outdir / "eeg_prism_fixed_cell_subjects.pdf")
    plt.close(fig)


def _write_report(
    *,
    region: str,
    window_label: str,
    wide: pd.DataFrame,
    contrast_summary: pd.DataFrame,
    permutation_summary: pd.DataFrame,
    search_control: pd.DataFrame,
    outpath: Path,
) -> None:
    rows = contrast_summary.set_index("contrast")
    perms = permutation_summary.set_index("contrast")
    n_subjects = int(rows.loc["auc_prism_gain_over_raw", "n_subjects"])
    n_trials = int(wide.shape[0])
    lines = [
        f"# Fixed Cell EEG PRISM Check: {region} {window_label} ms",
        "",
        f"Sample: {n_subjects} subjects, {n_trials} complete-case trials.",
        "The cell is analysed with subject as the unit of inference.",
        "",
        "## AUC gain over raw EEG",
        "",
    ]
    for name in ("auc_prism_gain_over_raw", "auc_var_gain_over_raw", "auc_prism_minus_var_hybrid"):
        row = rows.loc[name]
        extra = ""
        if name in perms.index:
            extra = f", permutation p={_format_p(float(perms.loc[name, 'perm_p_greater']))}"
        lines.append(
            f"- {name}: mean={row['mean']:.4f}, 95% CI [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}], "
            f"{int(row['positive_subjects'])}/{int(row['n_subjects'])} positive subjects, "
            f"paired t p={_format_p(float(row['ttest_p']))}, Wilcoxon p={_format_p(float(row['wilcoxon_p']))}, "
            f"sign p={_format_p(float(row['sign_p_greater']))}{extra}."
        )

    lines.extend(["", "## Hit-minus-miss contribution", ""])
    for name in (
        "prism_hit_minus_miss",
        "prism_stimamp_resid_hit_minus_miss",
        "prism_stimamp_matched_hit_minus_miss",
        "var_hit_minus_miss",
    ):
        row = rows.loc[name]
        extra = ""
        if name in perms.index:
            extra = f", permutation p={_format_p(float(perms.loc[name, 'perm_p_greater']))}"
        lines.append(
            f"- {name}: mean={row['mean']:.4f}, 95% CI [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}], "
            f"{int(row['positive_subjects'])}/{int(row['n_subjects'])} positive subjects, "
            f"paired t p={_format_p(float(row['ttest_p']))}, Wilcoxon p={_format_p(float(row['wilcoxon_p']))}, "
            f"sign p={_format_p(float(row['sign_p_greater']))}{extra}."
        )
    if not search_control.empty:
        row = search_control.iloc[0]
        lines.extend(
            [
                "",
                "## Search-space control",
                "",
                f"- PRISM contribution rank across {int(row['n_cells'])} region-window cells: "
                f"{int(row['target_rank_by_observed_t'])}.",
                f"- Max-stat permutation p={_format_p(float(row['max_stat_p_greater']))} "
                f"over {int(row['n_permutations'])} subject-preserving permutations.",
            ]
        )
    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    *,
    root: Path,
    outdir: Path,
    region: str,
    window_label: str,
    n_permutations: int,
    max_stat_permutations: int,
    seed: int,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    wide = _load_wide(root, region=region, window_label=window_label)
    subject_auc = _subject_auc_table(wide)
    contribution = _subject_contribution_table(wide)
    contrast_summary = _contrast_table(subject_auc, contribution)
    permutation = _permutation_summary(wide, contrast_summary, n_permutations=n_permutations, seed=seed)
    search_control = _max_stat_search_control(
        _load_scores(root),
        target_region=region,
        target_window=window_label,
        n_permutations=max_stat_permutations,
        seed=seed + 1,
    )

    subject_auc.to_csv(outdir / "fixed_cell_subject_auc.csv", index=False)
    contribution.to_csv(outdir / "fixed_cell_subject_contribution.csv", index=False)
    contrast_summary.to_csv(outdir / "fixed_cell_contrast_summary.csv", index=False)
    permutation.to_csv(outdir / "fixed_cell_permutation_summary.csv", index=False)
    search_control.to_csv(outdir / "fixed_cell_search_control.csv", index=False)
    _plot_subjects(subject_auc, contribution, outdir)
    _write_report(
        region=region,
        window_label=window_label,
        wide=wide,
        contrast_summary=contrast_summary,
        permutation_summary=permutation,
        search_control=search_control,
        outpath=outdir / "fixed_cell_summary.md",
    )
    print(f"Wrote {outdir / 'fixed_cell_summary.md'}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--window-label", default=DEFAULT_WINDOW)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--max-stat-permutations", type=int, default=DEFAULT_MAX_STAT_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    run(
        root=Path(args.root),
        outdir=Path(args.outdir),
        region=str(args.region),
        window_label=str(args.window_label),
        n_permutations=int(args.permutations),
        max_stat_permutations=int(args.max_stat_permutations),
        seed=int(args.seed),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
