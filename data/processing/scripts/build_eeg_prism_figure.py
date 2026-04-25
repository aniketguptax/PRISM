"""Build the EEG PRISM validation figure from summary tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO_ROOT / "data/results_prism/eeg_prism_central_full_pca_q4"
MODEL_ORDER = (
    "raw_central_delta",
    "raw_central_plus_prism_central_augmented",
    "raw_central_plus_baseline_central_augmented",
)
MODEL_LABELS = {
    "raw_central_delta": "Raw central\nEEG",
    "raw_central_plus_prism_central_augmented": "Raw EEG\n+ PRISM",
    "raw_central_plus_baseline_central_augmented": "Raw EEG\n+ VAR",
}
BLUE = "#1f5a8a"
TEAL = "#287c7c"
ORANGE = "#d46a1f"
RED = "#b3262e"
GREY = "#6f7780"
LIGHT_GREY = "#e4e7eb"
DARK = "#18212b"


def _format_p(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    if value < 1e-4:
        return f"{value:.1e}"
    if value < 0.01:
        return f"{value:.3f}"
    return f"{value:.2f}"


def _load_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _pairwise_delta(pairwise: pd.DataFrame, model_a: str, model_b: str) -> pd.Series:
    mask = (
        (pairwise["metric"] == "auc")
        & (pairwise["model_a"] == model_a)
        & (pairwise["model_b"] == model_b)
    )
    if mask.any():
        return pairwise.loc[mask].iloc[0]
    reverse = (
        (pairwise["metric"] == "auc")
        & (pairwise["model_a"] == model_b)
        & (pairwise["model_b"] == model_a)
    )
    if reverse.any():
        row = pairwise.loc[reverse].iloc[0].copy()
        row["mean_delta_model_b_minus_a"] = -float(row["mean_delta_model_b_minus_a"])
        return row
    raise ValueError(f"Could not find pairwise AUC contrast for {model_a} and {model_b}")


def panel_auc(ax: plt.Axes, group_summary: pd.DataFrame, pairwise: pd.DataFrame) -> None:
    rows = (
        group_summary.set_index("model_name")
        .loc[list(MODEL_ORDER)]
        .reset_index()
    )
    x = np.arange(len(rows), dtype=float)
    colours = [GREY, TEAL, ORANGE]
    ax.bar(
        x,
        rows["auc_mean"],
        color=colours,
        edgecolor=DARK,
        linewidth=0.7,
        width=0.62,
    )
    ax.errorbar(
        x,
        rows["auc_mean"],
        yerr=rows["auc_std"],
        fmt="none",
        ecolor=DARK,
        elinewidth=1.0,
        capsize=3,
    )
    ax.axhline(0.5, color=GREY, lw=0.9, ls=(0, (4, 3)))
    ax.set_ylim(0.5, 0.67)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[name] for name in rows["model_name"]], fontsize=8.5)
    ax.set_ylabel("Subject mean held-out AUC")
    ax.set_title("A  PRISM adds detection evidence", loc="left", fontweight="bold")
    ax.grid(True, axis="y", color=LIGHT_GREY, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)

    prism_delta = _pairwise_delta(
        pairwise,
        "raw_central_delta",
        "raw_central_plus_prism_central_augmented",
    )
    var_delta = _pairwise_delta(
        pairwise,
        "raw_central_delta",
        "raw_central_plus_baseline_central_augmented",
    )
    prism_vs_var = _pairwise_delta(
        pairwise,
        "raw_central_plus_baseline_central_augmented",
        "raw_central_plus_prism_central_augmented",
    )
    text = (
        f"Raw + PRISM gain: {float(prism_delta['mean_delta_model_b_minus_a']):+.3f}, "
        f"p={_format_p(float(prism_delta['ttest_p']))}\n"
        f"Raw + VAR gain: {float(var_delta['mean_delta_model_b_minus_a']):+.3f}, "
        f"p={_format_p(float(var_delta['ttest_p']))}\n"
        f"PRISM vs VAR: {float(prism_vs_var['mean_delta_model_b_minus_a']):+.3f}, "
        f"p={_format_p(float(prism_vs_var['ttest_p']))}"
    )
    ax.text(
        0.03,
        0.96,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor=LIGHT_GREY),
    )


def _spec_a_row(table: pd.DataFrame, term_prefix: str) -> pd.Series:
    mask = table["specification"].astype(str).str.startswith("A") & table[
        "term"
    ].astype(str).str.startswith(term_prefix)
    if not mask.any():
        raise ValueError(f"Could not find specification A term {term_prefix!r}")
    return table.loc[mask].iloc[0]


def panel_lme(ax: plt.Axes, detection_lme: pd.DataFrame, confidence_lme: pd.DataFrame) -> None:
    det = _spec_a_row(detection_lme, "sdt_int")
    conf = _spec_a_row(confidence_lme, "confidence")
    rows = pd.DataFrame(
        [
            {"label": "Detection\n(hit vs miss)", **det.to_dict()},
            {"label": "Confidence\n(hits only)", **conf.to_dict()},
        ]
    )
    y = np.arange(len(rows), dtype=float)
    err_lo = rows["beta"] - rows["ci95_low"]
    err_hi = rows["ci95_high"] - rows["beta"]
    ax.axvline(0.0, color=GREY, lw=0.9, ls=(0, (4, 3)))
    ax.errorbar(
        rows["beta"],
        y,
        xerr=[err_lo, err_hi],
        fmt="none",
        ecolor=DARK,
        elinewidth=1.1,
        capsize=3,
        zorder=2,
    )
    ax.scatter(
        rows["beta"],
        y,
        s=72,
        color=[TEAL, "#b9c0c8"],
        edgecolor=DARK,
        linewidth=0.7,
        zorder=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(rows["label"], fontsize=8.8)
    ax.invert_yaxis()
    ax.set_xlabel("PRISM contribution coefficient (95% CI)")
    ax.set_title("B  Trial-level mixed effects", loc="left", fontweight="bold")
    ax.grid(True, axis="x", color=LIGHT_GREY, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    xmin = min(float(rows["ci95_low"].min()), -0.004)
    xmax = max(float(rows["ci95_high"].max()), 0.036)
    ax.set_xlim(xmin - 0.004, xmax + 0.004)

    for _, row in rows.iterrows():
        ax.text(
            float(row["ci95_high"]) + 0.002,
            float(row.name),
            f"p={_format_p(float(row['p']))}",
            va="center",
            ha="left",
            fontsize=8.2,
            color=DARK,
        )


def panel_subjects(ax: plt.Axes, per_subject: pd.DataFrame, summary: pd.DataFrame) -> None:
    det = per_subject["spearman_partial_pred_fit_vs_detection"].to_numpy(dtype=float)
    conf = per_subject["spearman_partial_pred_fit_vs_confidence"].to_numpy(dtype=float)
    x_det = np.full_like(det, 0.0, dtype=float)
    x_conf = np.full_like(conf, 1.0, dtype=float)
    rng = np.random.default_rng(0)
    jitter = rng.normal(0.0, 0.035, size=det.size)

    for idx in range(det.size):
        ax.plot([x_det[idx] + jitter[idx], x_conf[idx] + jitter[idx]], [det[idx], conf[idx]], color="#c9ced6", lw=0.75, zorder=1)
    ax.scatter(x_det + jitter, det, s=34, color=TEAL, edgecolor=DARK, linewidth=0.45, zorder=3)
    ax.scatter(x_conf + jitter, conf, s=34, color="#b9c0c8", edgecolor=DARK, linewidth=0.45, zorder=3)
    ax.axhline(0.0, color=GREY, lw=0.9, ls=(0, (4, 3)))
    means = [float(np.mean(det)), float(np.mean(conf))]
    ax.scatter([0, 1], means, s=120, marker="D", color=[TEAL, "#7e8791"], edgecolor="white", linewidth=0.8, zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Detection", "Confidence"], fontsize=9)
    ax.set_ylabel("Subject partial Spearman rho")
    ax.set_title("C  Subject-level partial associations", loc="left", fontweight="bold")
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(min(-0.14, float(np.nanmin([det.min(), conf.min()])) - 0.03), max(0.27, float(np.nanmax([det.max(), conf.max()])) + 0.03))
    ax.grid(True, axis="y", color=LIGHT_GREY, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)

    det_row = summary.loc[summary["test"] == "detection_rho_vs_zero"].iloc[0]
    conf_row = summary.loc[summary["test"] == "confidence_rho_vs_zero"].iloc[0]
    text = (
        f"Detection mean rho={float(det_row['mean']):+.3f}, "
        f"Holm p={_format_p(float(det_row['ttest_p_holm']))}\n"
        f"Confidence mean rho={float(conf_row['mean']):+.3f}, "
        f"Holm p={_format_p(float(conf_row['ttest_p_holm']))}"
    )
    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=LIGHT_GREY),
    )


def build_figure(root: Path, outfile: Path) -> dict[str, float]:
    evidence_dir = root / "summary_evidence"
    mech_dir = root / "summary_mechanistic"
    group_summary = _load_required(evidence_dir / "evidence_group_summary.csv")
    pairwise = _load_required(evidence_dir / "evidence_pairwise_summary.csv")
    detection_lme = _load_required(mech_dir / "detection_mixed_effects.csv")
    confidence_lme = _load_required(mech_dir / "confidence_mixed_effects.csv")
    per_subject = _load_required(mech_dir / "per_subject_spearman_partial.csv")
    spearman_summary = _load_required(mech_dir / "spearman_dissociation_summary.csv")

    fig = plt.figure(figsize=(10.5, 6.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0], height_ratios=[1.0, 1.0])
    panel_auc(fig.add_subplot(gs[:, 0]), group_summary, pairwise)
    panel_lme(fig.add_subplot(gs[0, 1]), detection_lme, confidence_lme)
    panel_subjects(fig.add_subplot(gs[1, 1]), per_subject, spearman_summary)
    fig.suptitle(
        "Continuous PRISM recovers detection-linked predictive state structure in EEG",
        fontsize=13.5,
        fontweight="bold",
        x=0.5,
        y=1.02,
    )
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    prism_delta = _pairwise_delta(
        pairwise,
        "raw_central_delta",
        "raw_central_plus_prism_central_augmented",
    )
    var_delta = _pairwise_delta(
        pairwise,
        "raw_central_delta",
        "raw_central_plus_baseline_central_augmented",
    )
    prism_vs_var = _pairwise_delta(
        pairwise,
        "raw_central_plus_baseline_central_augmented",
        "raw_central_plus_prism_central_augmented",
    )
    det = _spec_a_row(detection_lme, "sdt_int")
    conf = _spec_a_row(confidence_lme, "confidence")
    return {
        "raw_plus_prism_auc_gain": float(prism_delta["mean_delta_model_b_minus_a"]),
        "raw_plus_prism_auc_p": float(prism_delta["ttest_p"]),
        "raw_plus_var_auc_gain": float(var_delta["mean_delta_model_b_minus_a"]),
        "raw_plus_var_auc_p": float(var_delta["ttest_p"]),
        "prism_minus_var_auc": float(prism_vs_var["mean_delta_model_b_minus_a"]),
        "prism_minus_var_auc_p": float(prism_vs_var["ttest_p"]),
        "detection_beta": float(det["beta"]),
        "detection_p": float(det["p"]),
        "confidence_beta": float(conf["beta"]),
        "confidence_p": float(conf["p"]),
        "n_trials": int(det["n_obs"]),
        "n_subjects": int(det["n_groups"]),
    }


def write_caption(numbers: dict[str, float], outfile: Path) -> None:
    caption = (
        "# EEG PRISM Figure Caption\n\n"
        "Continuous PRISM predictive-state summaries extracted from the pre-stimulus "
        "central EEG window add significant hit-versus-miss evidence beyond the raw "
        "central evoked response. Raw EEG plus PRISM improved held-out AUC by "
        f"{numbers['raw_plus_prism_auc_gain']:+.3f} over raw EEG alone "
        f"(paired p={_format_p(numbers['raw_plus_prism_auc_p'])}), while the stronger "
        "raw EEG plus VAR predictive-fit model improved AUC by "
        f"{numbers['raw_plus_var_auc_gain']:+.3f} "
        f"(paired p={_format_p(numbers['raw_plus_var_auc_p'])}). Trial-level mixed "
        "effects confirmed that the PRISM contribution was larger on hit trials than "
        f"miss trials (beta={numbers['detection_beta']:+.4f}, "
        f"p={_format_p(numbers['detection_p'])}, n={int(numbers['n_trials'])} trials, "
        f"{int(numbers['n_subjects'])} subjects), whereas the confidence association "
        f"on hits was weak (beta={numbers['confidence_beta']:+.4f}, "
        f"p={_format_p(numbers['confidence_p'])}). PRISM therefore provides a positive "
        "EEG validation of detection-linked predictive state structure, but the VAR "
        "predictive-fit baseline remains the stronger EEG evidence model."
    )
    outfile.write_text(caption + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--outfile",
        type=Path,
        default=DEFAULT_ROOT / "summary_mechanistic/eeg_prism_headline.png",
    )
    args = parser.parse_args()

    numbers = build_figure(args.root, args.outfile)
    pd.DataFrame([numbers]).to_csv(
        args.outfile.with_name("eeg_prism_headline_key_numbers.csv"),
        index=False,
    )
    write_caption(numbers, args.outfile.with_name("eeg_prism_headline_caption.md"))
    print(f"Wrote {args.outfile}")
    print(f"Wrote {args.outfile.with_suffix('.pdf')}")
    print(f"Wrote {args.outfile.with_name('eeg_prism_headline_key_numbers.csv')}")
    print(f"Wrote {args.outfile.with_name('eeg_prism_headline_caption.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
