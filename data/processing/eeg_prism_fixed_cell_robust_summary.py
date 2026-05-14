from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from eeg_subject_decoder import (
    DEFAULT_BASELINE_END_MS,
    DEFAULT_BASELINE_RESULTS,
    DEFAULT_BASELINE_START_MS,
    DEFAULT_DERIVATIVES_DIR,
    DEFAULT_EXPORT_DIR,
    _ci95,
    _format_p,
    run as run_subject_decoder,
)


DEFAULT_ROOT = Path("data/results_prism/eeg_prism_fixed_cell_robust")
DEFAULT_OUTDIR = DEFAULT_ROOT / "summary"
DEFAULT_REGION = "occipital"
DEFAULT_TEST_START_MS = 125.0
DEFAULT_TEST_END_MS = 375.0
DEFAULT_BASELINE_REP_DIM = 4
SETTING_RE = re.compile(r"^pca_d(?P<rep_dim>\d+)_(?P<builder>.+)_eps(?P<eps>\d+)$")


def _decode_eps(raw: str) -> float:
    if len(raw) == 1:
        return float(raw)
    return float(f"{raw[:-2]}.{raw[-2:]}")


def _setting_from_path(path: Path) -> dict[str, object] | None:
    match = SETTING_RE.match(path.name)
    if match is None:
        return None
    return {
        "setting": path.name,
        "projection_mode": "pca",
        "rep_dim": int(match.group("rep_dim")),
        "macro_builder": match.group("builder"),
        "macro_eps": _decode_eps(match.group("eps")),
    }


def _one_sample(values: pd.Series | np.ndarray) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    ci_low, ci_high = _ci95(arr)
    ttest = stats.ttest_1samp(arr, 0.0) if arr.size >= 2 else None
    try:
        wilcoxon_p = float(stats.wilcoxon(arr).pvalue)
    except ValueError:
        wilcoxon_p = math.nan
    positive = int(np.sum(arr > 0.0))
    sign_p = (
        float(stats.binomtest(positive, arr.size, p=0.5, alternative="greater").pvalue)
        if arr.size
        else math.nan
    )
    return {
        "n_subjects": int(arr.size),
        "mean": float(arr.mean()) if arr.size else math.nan,
        "median": float(np.median(arr)) if arr.size else math.nan,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "positive_subjects": positive,
        "ttest_p": float(ttest.pvalue) if ttest is not None else math.nan,
        "wilcoxon_p": wilcoxon_p,
        "sign_p_greater": sign_p,
    }


def _wide_scores(scores: pd.DataFrame) -> pd.DataFrame:
    index_cols = ["subject", "trial_idx", "label", "sdt", "confidence", "stimamp"]
    return scores.pivot_table(index=index_cols, columns="model_name", values="decoder_score").reset_index()


def _contribution_by_subject(scores: pd.DataFrame) -> pd.DataFrame:
    wide = _wide_scores(scores)
    rows: list[dict[str, object]] = []
    for subject, group in wide.groupby("subject", sort=True):
        labels = group["label"].to_numpy(dtype=int)
        raw = group["raw_central_delta"].to_numpy(dtype=float)
        prism = group["raw_plus_prism"].to_numpy(dtype=float)
        contrib = prism - raw
        hit = contrib[labels == 1]
        miss = contrib[labels == 0]
        hit_rows = group.loc[labels == 1, ["confidence"]].copy()
        hit_rows["contrib"] = hit
        hit_rows = hit_rows.dropna()
        confidence_rho = math.nan
        if hit_rows.shape[0] >= 5:
            confidence_rho = float(stats.spearmanr(hit_rows["contrib"], hit_rows["confidence"]).statistic)
        rows.append(
            {
                "subject": subject,
                "prism_hit_minus_miss": float(hit.mean() - miss.mean()),
                "prism_confidence_rho_hits": confidence_rho,
            }
        )
    return pd.DataFrame(rows)


def _summarise_setting(setting: dict[str, object], summary_dir: Path) -> tuple[dict[str, object], pd.DataFrame]:
    subject_summary = pd.read_csv(summary_dir / "eeg_subject_decoder_subject_summary.csv")
    pairwise = pd.read_csv(summary_dir / "eeg_subject_decoder_pairwise.csv")
    scores = pd.read_csv(summary_dir / "eeg_subject_decoder_trial_scores.csv")

    wide_auc = subject_summary.pivot(index="subject", columns="model_name", values="auc").reset_index()
    wide_auc["auc_prism_gain_over_raw"] = wide_auc["raw_plus_prism"] - wide_auc["raw_central_delta"]
    wide_auc["auc_var_gain_over_raw"] = wide_auc["raw_plus_var"] - wide_auc["raw_central_delta"]
    contrib = _contribution_by_subject(scores)
    values = wide_auc.merge(contrib, on="subject", how="inner", validate="one_to_one")

    row: dict[str, object] = dict(setting)
    for metric in (
        "auc_prism_gain_over_raw",
        "auc_var_gain_over_raw",
        "prism_hit_minus_miss",
        "prism_confidence_rho_hits",
    ):
        stats_row = _one_sample(values[metric])
        for key, value in stats_row.items():
            row[f"{metric}_{key}"] = value
    prism_pair = pairwise.loc[pairwise["contrast"].eq("prism_gain_over_raw")]
    if not prism_pair.empty:
        row["pairwise_prism_gain_p"] = float(prism_pair.iloc[0]["ttest_p"])
    values.insert(0, "setting", str(setting["setting"]))
    return row, values


def _write_report(summary: pd.DataFrame, outpath: Path) -> None:
    if summary.empty:
        outpath.write_text("# Fixed-Cell Robustness Summary\n\nNo settings were available.\n", encoding="utf-8")
        return
    n_settings = int(summary.shape[0])
    positive_auc = int((summary["auc_prism_gain_over_raw_mean"] > 0.0).sum())
    positive_contrib = int((summary["prism_hit_minus_miss_mean"] > 0.0).sum())
    positive_conf = int((summary["prism_confidence_rho_hits_mean"] > 0.0).sum())
    all_subject_contrib = int((summary["prism_hit_minus_miss_positive_subjects"] == 18).sum())
    all_subject_conf = int((summary["prism_confidence_rho_hits_positive_subjects"] == 18).sum())
    sig_auc = int((summary["auc_prism_gain_over_raw_ttest_p"] < 0.05).sum())
    sig_contrib = int((summary["prism_hit_minus_miss_ttest_p"] < 0.05).sum())
    sig_conf = int((summary["prism_confidence_rho_hits_ttest_p"] < 0.05).sum())
    original = summary.loc[
        summary["rep_dim"].eq(4)
        & summary["macro_builder"].eq("hierarchical_complete")
        & np.isclose(summary["macro_eps"], 0.25)
    ]
    best = summary.sort_values("prism_hit_minus_miss_ttest_p").iloc[0]

    lines = [
        "# Fixed-Cell Robustness Summary",
        "",
        f"Settings completed: {n_settings}.",
        f"PRISM AUC gain is positive in {positive_auc}/{n_settings} settings and p<0.05 in {sig_auc}/{n_settings}.",
        f"PRISM hit-minus-miss contribution is positive in {positive_contrib}/{n_settings} settings and p<0.05 in {sig_contrib}/{n_settings}.",
        f"The contribution is positive in all 18 subjects for {all_subject_contrib}/{n_settings} settings.",
        f"PRISM confidence-on-hit rho is positive in {positive_conf}/{n_settings} settings and p<0.05 in {sig_conf}/{n_settings}.",
        f"The confidence rho is positive in all 18 subjects for {all_subject_conf}/{n_settings} settings.",
        "",
        "## Original setting",
        "",
    ]
    if original.empty:
        lines.append("The original setting was not present in this grid.")
    else:
        row = original.iloc[0]
        lines.append(
            f"- d=4, hierarchical_complete, eps=0.25: AUC gain={row['auc_prism_gain_over_raw_mean']:.4f}, "
            f"contribution={row['prism_hit_minus_miss_mean']:.4f}, "
            f"contribution p={_format_p(float(row['prism_hit_minus_miss_ttest_p']))}, "
            f"confidence rho={row['prism_confidence_rho_hits_mean']:.4f}, "
            f"confidence p={_format_p(float(row['prism_confidence_rho_hits_ttest_p']))}."
        )
    lines.extend(
        [
            "",
            "## Strongest setting by contribution p-value",
            "",
            f"- {best['setting']}: AUC gain={best['auc_prism_gain_over_raw_mean']:.4f}, "
            f"contribution={best['prism_hit_minus_miss_mean']:.4f}, "
            f"{int(best['prism_hit_minus_miss_positive_subjects'])}/18 positive subjects, "
            f"contribution p={_format_p(float(best['prism_hit_minus_miss_ttest_p']))}.",
        ]
    )
    best_conf = summary.sort_values("prism_confidence_rho_hits_ttest_p").iloc[0]
    lines.extend(
        [
            "",
            "## Strongest setting by confidence p-value",
            "",
            f"- {best_conf['setting']}: AUC gain={best_conf['auc_prism_gain_over_raw_mean']:.4f}, "
            f"confidence rho={best_conf['prism_confidence_rho_hits_mean']:.4f}, "
            f"{int(best_conf['prism_confidence_rho_hits_positive_subjects'])}/18 positive subjects, "
            f"confidence p={_format_p(float(best_conf['prism_confidence_rho_hits_ttest_p']))}.",
        ]
    )
    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    *,
    root: Path,
    outdir: Path,
    baseline_results_dir: Path,
    derivatives_dir: Path,
    export_dir: Path,
    region: str,
    test_start_ms: float,
    test_end_ms: float,
    baseline_rep_dim: int,
    n_folds: int,
    ridge: float,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    subject_rows: list[pd.DataFrame] = []
    for setting_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        setting = _setting_from_path(setting_dir)
        if setting is None:
            continue
        summary_dir = outdir / str(setting["setting"])
        if not (summary_dir / "eeg_subject_decoder_subject_summary.csv").exists():
            run_subject_decoder(
                baseline_results_dir=baseline_results_dir,
                prism_results_dir=setting_dir,
                derivatives_dir=derivatives_dir,
                export_dir=export_dir,
                outdir=summary_dir,
                region=region,
                rep_dim=int(setting["rep_dim"]),
                baseline_rep_dim=baseline_rep_dim,
                test_start_ms=test_start_ms,
                test_end_ms=test_end_ms,
                baseline_start_ms=DEFAULT_BASELINE_START_MS,
                baseline_end_ms=DEFAULT_BASELINE_END_MS,
                prism_model_family="prism_pca",
                positive_sdt="hit",
                negative_sdt="miss",
                n_folds=n_folds,
                ridge=ridge,
            )
        row, values = _summarise_setting(setting, summary_dir)
        rows.append(row)
        subject_rows.append(values)

    summary = pd.DataFrame(rows).sort_values(["rep_dim", "macro_eps", "macro_builder"])
    summary.to_csv(outdir / "fixed_cell_robust_setting_summary.csv", index=False)
    if subject_rows:
        pd.concat(subject_rows, ignore_index=True).to_csv(
            outdir / "fixed_cell_robust_subject_values.csv",
            index=False,
        )
    _write_report(summary, outdir / "fixed_cell_robust_summary.md")
    print(f"Wrote {outdir / 'fixed_cell_robust_summary.md'}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--baseline-results-dir", type=Path, default=DEFAULT_BASELINE_RESULTS)
    parser.add_argument("--derivatives-dir", type=Path, default=DEFAULT_DERIVATIVES_DIR)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--test-start-ms", type=float, default=DEFAULT_TEST_START_MS)
    parser.add_argument("--test-end-ms", type=float, default=DEFAULT_TEST_END_MS)
    parser.add_argument("--baseline-rep-dim", type=int, default=DEFAULT_BASELINE_REP_DIM)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=1e-3)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    run(
        root=Path(args.root),
        outdir=Path(args.outdir),
        baseline_results_dir=Path(args.baseline_results_dir),
        derivatives_dir=Path(args.derivatives_dir),
        export_dir=Path(args.export_dir),
        region=str(args.region),
        test_start_ms=float(args.test_start_ms),
        test_end_ms=float(args.test_end_ms),
        baseline_rep_dim=int(args.baseline_rep_dim),
        n_folds=int(args.n_folds),
        ridge=float(args.ridge),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
