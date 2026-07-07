#!/usr/bin/env python3
"""
D1-D4 morphological recovery analysis and figure generation.

Implements the methods described in Section 3.6.2:
  (9)  empirical first Wasserstein distance for equal-size samples
  (10) normalized Wasserstein distance using the D1 IQR
  (11) paired label-swap permutation p-value
  (12) normalized mean absolute error (NMAE)
  (13) Lin's concordance correlation coefficient (CCC)
  (14) paired difference d = D4 - D1
  (15) matched-pairs rank-biserial correlation
  (16) per-case standardized absolute error
  (17) predefined vertical and planar group mean errors

Expected workbook structure:
  - one D1 sheet and one D4 sheet
  - a shared unique case_id column
  - columns: AveH, MaxH, StdH, TPI, BCR, LEI, CPR, VCI, FAR

Example:
  python d1_d4_morphological_recovery.py \
      --input-xlsx Data_S1_morphological_parameters.xlsx \
      --d1-sheet D1_internal_100 \
      --d4-sheet D4_internal_100 \
      --output-dir outputs

Dependencies:
  numpy, pandas, scipy, matplotlib, openpyxl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import scipy
from scipy import stats


METRIC_INFO: Dict[str, Dict[str, str]] = {
    "AveH": {"label": "Average Height", "label_unit": "Average Height (m)", "group": "Vertical", "unit": "m"},
    "MaxH": {"label": "Max Height", "label_unit": "Max Height (m)", "group": "Vertical", "unit": "m"},
    "StdH": {"label": "Height SD", "label_unit": "Height SD (m)", "group": "Vertical", "unit": "m"},
    "TPI":  {"label": "Tower–Podium Index", "label_unit": "Tower–Podium Index", "group": "Vertical", "unit": "-"},
    "BCR":  {"label": "Building Coverage Ratio", "label_unit": "Building Coverage Ratio", "group": "Planar", "unit": "-"},
    "LEI":  {"label": "Linear Extension Index", "label_unit": "Linear Extension Index", "group": "Planar", "unit": "-"},
    "CPR":  {"label": "Central Plaza Ratio", "label_unit": "Central Plaza Ratio", "group": "Planar", "unit": "-"},
    "VCI":  {"label": "Volume Concentration Index", "label_unit": "Volume Conc. Index", "group": "Planar", "unit": "-"},
    "FAR":  {"label": "Floor Area Ratio", "label_unit": "Floor Area Ratio (FAR)", "group": "Hybrid", "unit": "-"},
}

METRIC_ORDER = ["AveH", "MaxH", "StdH", "TPI", "BCR", "LEI", "CPR", "VCI", "FAR"]
VERTICAL_METRICS = ["AveH", "MaxH", "StdH", "TPI"]
PLANAR_METRICS = ["BCR", "LEI", "CPR", "VCI"]

# Colors chosen to reproduce the visual logic of the supplied figures.
COLOR_OVER = "#D66A6A"
COLOR_UNDER = "#5A88BF"
COLOR_NMAE = "#8E9BB3"
COLOR_CCC_GOOD = "#45A79F"
COLOR_CCC_MID = "#E5C66B"
COLOR_NEUTRAL = "#B7B7B7"
COLOR_PURPLE = "#A77AC3"
COLOR_ERROR = "#284866"
GRID_COLOR = "#D9D9D9"


@dataclass(frozen=True)
class AnalysisConfig:
    n_bootstrap: int = 10_000
    n_permutations: int = 10_000
    random_seed: int = 42
    ci_level: float = 0.95
    tpi_tolerance: float = 1e-3
    dpi: int = 300


def percentile_ci(values: np.ndarray, level: float = 0.95) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (np.nan, np.nan)
    alpha = (1.0 - level) / 2.0
    return tuple(np.quantile(values, [alpha, 1.0 - alpha]))


def iqr(values: np.ndarray, axis=None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.percentile(values, 75, axis=axis) - np.percentile(values, 25, axis=axis)


def empirical_w1_equal_n(x: np.ndarray, y: np.ndarray, axis=None) -> np.ndarray:
    """Equation (9): W1 = mean absolute difference between ordered samples."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.mean(np.abs(np.sort(x, axis=axis) - np.sort(y, axis=axis)), axis=axis)


def lin_ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Equation (13), using sample variances and sample covariance (ddof=1)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return np.nan
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    cov = np.cov(x, y, ddof=1)[0, 1]
    denom = vx + vy + (np.mean(x) - np.mean(y)) ** 2
    return float(2.0 * cov / denom) if denom > 0 else np.nan


def vectorized_ccc(xb: np.ndarray, yb: np.ndarray) -> np.ndarray:
    n = xb.shape[1]
    mx = xb.mean(axis=1)
    my = yb.mean(axis=1)
    vx = xb.var(axis=1, ddof=1)
    vy = yb.var(axis=1, ddof=1)
    cov = ((xb - mx[:, None]) * (yb - my[:, None])).sum(axis=1) / (n - 1)
    denom = vx + vy + (mx - my) ** 2
    return np.divide(2.0 * cov, denom, out=np.full_like(denom, np.nan), where=denom > 0)


def matched_rank_biserial_from_diff(diff: np.ndarray) -> float:
    """Equation (15): (sum positive ranks - sum negative ranks) / total ranks."""
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff) & (diff != 0)]
    if diff.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(diff), method="average")
    s_pos = ranks[diff > 0].sum()
    s_neg = ranks[diff < 0].sum()
    denom = s_pos + s_neg
    return float((s_pos - s_neg) / denom) if denom > 0 else 0.0


def vectorized_rank_biserial(diff_boot: np.ndarray) -> np.ndarray:
    """
    Rank-biserial correlation for each bootstrap row.
    Zeros are omitted from ranking, matching scipy.stats.wilcoxon(zero_method='wilcox').
    """
    abs_diff = np.abs(diff_boot).astype(float)
    abs_diff[diff_boot == 0] = np.nan
    ranks = stats.rankdata(abs_diff, axis=1, method="average", nan_policy="omit")
    ranks = np.nan_to_num(ranks, nan=0.0)
    s_pos = np.sum(np.where(diff_boot > 0, ranks, 0.0), axis=1)
    s_neg = np.sum(np.where(diff_boot < 0, ranks, 0.0), axis=1)
    denom = s_pos + s_neg
    return np.divide(s_pos - s_neg, denom, out=np.zeros_like(denom), where=denom > 0)


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted_sorted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * p[idx]
        running_max = max(running_max, candidate)
        adjusted_sorted[rank] = min(1.0, running_max)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def significance_label(p: float) -> str:
    if not np.isfinite(p):
        return "n.s."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def format_p(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    if p < 0.0001:
        return f"{p:.2e}"
    if p < 0.001:
        return f"{p:.7f}".rstrip("0")
    return f"{p:.4f}".rstrip("0").rstrip(".")


def load_and_match(
    input_xlsx: Path,
    d1_sheet: str,
    d4_sheet: str,
    case_id_col: str,
) -> pd.DataFrame:
    d1 = pd.read_excel(input_xlsx, sheet_name=d1_sheet)
    d4 = pd.read_excel(input_xlsx, sheet_name=d4_sheet)

    required = {case_id_col, *METRIC_ORDER}
    for name, df in [("D1", d1), ("D4", d4)]:
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError(f"{name} sheet is missing columns: {missing}")
        if df[case_id_col].isna().any():
            raise ValueError(f"{name} sheet contains missing {case_id_col} values.")
        if df[case_id_col].duplicated().any():
            dup = df.loc[df[case_id_col].duplicated(), case_id_col].tolist()
            raise ValueError(f"{name} sheet contains duplicate case IDs: {dup[:10]}")

    d1_ids = set(d1[case_id_col].astype(str))
    d4_ids = set(d4[case_id_col].astype(str))
    if d1_ids != d4_ids:
        only_d1 = sorted(d1_ids - d4_ids)
        only_d4 = sorted(d4_ids - d1_ids)
        raise ValueError(
            "D1 and D4 case IDs do not match.\n"
            f"Only in D1: {only_d1[:10]}\nOnly in D4: {only_d4[:10]}"
        )

    d1 = d1.copy()
    d1["_input_order"] = np.arange(len(d1))
    merged = d1[[case_id_col, "_input_order", *METRIC_ORDER]].merge(
        d4[[case_id_col, *METRIC_ORDER]],
        on=case_id_col,
        how="inner",
        suffixes=("_D1", "_D4"),
        validate="one_to_one",
        sort=False,
    ).sort_values("_input_order").drop(columns="_input_order").reset_index(drop=True)

    for metric in METRIC_ORDER:
        for suffix in ("D1", "D4"):
            col = f"{metric}_{suffix}"
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
            if merged[col].isna().any() or not np.isfinite(merged[col]).all():
                bad = merged.loc[~np.isfinite(merged[col]), case_id_col].tolist()
                raise ValueError(f"Non-finite values found in {col}, cases: {bad[:10]}")
        if iqr(merged[f"{metric}_D1"].to_numpy()) <= 0:
            raise ValueError(f"D1 IQR is zero for {metric}; normalized statistics are undefined.")

    return merged


def check_tpi_consistency(
    matched: pd.DataFrame,
    case_id_col: str,
    tolerance: float,
    output_dir: Path,
) -> None:
    records = []
    for domain in ("D1", "D4"):
        maxh = matched[f"MaxH_{domain}"].to_numpy(float)
        aveh = matched[f"AveH_{domain}"].to_numpy(float)
        stored = matched[f"TPI_{domain}"].to_numpy(float)
        calculated = np.divide(maxh - aveh, maxh, out=np.full_like(maxh, np.nan), where=maxh != 0)
        delta = stored - calculated
        flagged = np.abs(delta) > tolerance
        for idx in np.where(flagged)[0]:
            records.append({
                "case_id": matched.loc[idx, case_id_col],
                "dataset": domain,
                "MaxH": maxh[idx],
                "AveH": aveh[idx],
                "TPI_stored": stored[idx],
                "TPI_from_definition": calculated[idx],
                "stored_minus_calculated": delta[idx],
            })

    if records:
        out = pd.DataFrame(records)
        out.to_csv(output_dir / "TPI_consistency_check.csv", index=False)
        warnings.warn(
            f"{len(out)} TPI values differ from (MaxH-AveH)/MaxH by more than "
            f"{tolerance:g}. Stored TPI values were used; see TPI_consistency_check.csv.",
            RuntimeWarning,
        )


def analyze_metrics(
    matched: pd.DataFrame,
    case_id_col: str,
    config: AnalysisConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    n = len(matched)
    rng_boot = np.random.default_rng(config.random_seed)
    boot_idx = rng_boot.integers(0, n, size=(config.n_bootstrap, n))

    # The same label-swap masks are used for every metric.
    rng_perm = np.random.default_rng(config.random_seed + 1)
    swap_masks = rng_perm.random((config.n_permutations, n)) < 0.5

    metric_rows = []
    wasserstein_rows = []
    standardized_errors: Dict[str, np.ndarray] = {}
    raw_wilcoxon_p = []
    raw_permutation_p = []

    for metric in METRIC_ORDER:
        x = matched[f"{metric}_D1"].to_numpy(float)
        y = matched[f"{metric}_D4"].to_numpy(float)
        diff = y - x
        d1_iqr = float(iqr(x))

        # Point estimates: equations (9), (10), (12), (13), and (15).
        raw_w1 = float(empirical_w1_equal_n(x, y))
        normalized_w1 = raw_w1 / d1_iqr
        mae = float(np.mean(np.abs(diff)))
        nmae = mae / d1_iqr
        ccc = lin_ccc(x, y)
        r_rb = matched_rank_biserial_from_diff(diff)

        standardized_errors[metric] = np.abs(diff) / d1_iqr

        # Pair-preserving bootstrap. Every statistic is recomputed, including IQR(D1).
        xb = x[boot_idx]
        yb = y[boot_idx]
        diffb = yb - xb
        iqr_b = iqr(xb, axis=1)
        valid_iqr = iqr_b > 0

        mae_b = np.mean(np.abs(diffb), axis=1)
        nmae_b = np.divide(mae_b, iqr_b, out=np.full_like(mae_b, np.nan), where=valid_iqr)
        ccc_b = vectorized_ccc(xb, yb)
        rrb_b = vectorized_rank_biserial(diffb)
        raw_w1_b = empirical_w1_equal_n(xb, yb, axis=1)
        nw1_b = np.divide(raw_w1_b, iqr_b, out=np.full_like(raw_w1_b, np.nan), where=valid_iqr)

        mae_ci = percentile_ci(mae_b, config.ci_level)
        nmae_ci = percentile_ci(nmae_b, config.ci_level)
        ccc_ci = percentile_ci(ccc_b, config.ci_level)
        rrb_ci = percentile_ci(rrb_b, config.ci_level)
        raw_w1_ci = percentile_ci(raw_w1_b, config.ci_level)
        nw1_ci = percentile_ci(nw1_b, config.ci_level)

        # Paired Wilcoxon signed-rank test for systematic directional shift.
        if np.all(diff == 0):
            wilcoxon_w, wilcoxon_p = 0.0, 1.0
        else:
            wres = stats.wilcoxon(
                diff,
                zero_method="wilcox",
                correction=False,
                alternative="two-sided",
                method="approx",
            )
            wilcoxon_w, wilcoxon_p = float(wres.statistic), float(wres.pvalue)
        raw_wilcoxon_p.append(wilcoxon_p)

        # Paired label-swap permutation test for normalized W1.
        xp = np.where(swap_masks, y[None, :], x[None, :])
        yp = np.where(swap_masks, x[None, :], y[None, :])
        perm_iqr = iqr(xp, axis=1)
        perm_w1 = empirical_w1_equal_n(xp, yp, axis=1)
        perm_nw1 = np.divide(
            perm_w1,
            perm_iqr,
            out=np.full_like(perm_w1, np.nan),
            where=perm_iqr > 0,
        )
        valid_perm = perm_nw1[np.isfinite(perm_nw1)]
        permutation_p = (1.0 + np.sum(valid_perm >= normalized_w1)) / (len(valid_perm) + 1.0)
        raw_permutation_p.append(float(permutation_p))

        metric_rows.append({
            "Metric": metric,
            "Label": METRIC_INFO[metric]["label_unit"],
            "Group": METRIC_INFO[metric]["group"],
            "n": n,
            "D1_Q1": np.percentile(x, 25),
            "D1_median": np.median(x),
            "D1_Q3": np.percentile(x, 75),
            "D4_Q1": np.percentile(y, 25),
            "D4_median": np.median(y),
            "D4_Q3": np.percentile(y, 75),
            "IQR_D1": d1_iqr,
            "MAE": mae,
            "MAE_CI_low": mae_ci[0],
            "MAE_CI_high": mae_ci[1],
            "NMAE": nmae,
            "NMAE_percent": nmae * 100.0,
            "NMAE_CI_low": nmae_ci[0],
            "NMAE_CI_high": nmae_ci[1],
            "NMAE_CI_low_percent": nmae_ci[0] * 100.0,
            "NMAE_CI_high_percent": nmae_ci[1] * 100.0,
            "CCC": ccc,
            "CCC_CI_low": ccc_ci[0],
            "CCC_CI_high": ccc_ci[1],
            "Wilcoxon_W": wilcoxon_w,
            "Wilcoxon_raw_p": wilcoxon_p,
            "paired_rank_biserial": r_rb,
            "rank_biserial_CI_low": rrb_ci[0],
            "rank_biserial_CI_high": rrb_ci[1],
        })

        wasserstein_rows.append({
            "Metric": metric,
            "Label": METRIC_INFO[metric]["label"],
            "Group": METRIC_INFO[metric]["group"],
            "n_D1": n,
            "n_D4": n,
            "IQR_D1": d1_iqr,
            "Raw_Wasserstein_W1": raw_w1,
            "Raw_W1_CI_low": raw_w1_ci[0],
            "Raw_W1_CI_high": raw_w1_ci[1],
            "Normalized_W1": normalized_w1,
            "Normalized_W1_CI_low": nw1_ci[0],
            "Normalized_W1_CI_high": nw1_ci[1],
            "Paired_permutation_p": permutation_p,
        })

    metric_stats = pd.DataFrame(metric_rows)
    wasserstein_stats = pd.DataFrame(wasserstein_rows)

    metric_stats["Wilcoxon_Holm_p"] = holm_adjust(raw_wilcoxon_p)
    metric_stats["Wilcoxon_Holm_sig"] = metric_stats["Wilcoxon_Holm_p"].map(significance_label)
    wasserstein_stats["Permutation_Holm_p"] = holm_adjust(raw_permutation_p)
    wasserstein_stats["Permutation_Holm_sig"] = wasserstein_stats["Permutation_Holm_p"].map(significance_label)

    # Equations (16) and (17): per-case standardized errors and predefined groups.
    case_errors = pd.DataFrame({case_id_col: matched[case_id_col].astype(str)})
    for metric in METRIC_ORDER:
        case_errors[f"e_{metric}"] = standardized_errors[metric]
    case_errors["E_vertical"] = case_errors[[f"e_{m}" for m in VERTICAL_METRICS]].mean(axis=1)
    case_errors["E_planar"] = case_errors[[f"e_{m}" for m in PLANAR_METRICS]].mean(axis=1)
    case_errors["Difference_vertical_minus_planar"] = case_errors["E_vertical"] - case_errors["E_planar"]

    pv_diff = case_errors["Difference_vertical_minus_planar"].to_numpy(float)
    pv_w = stats.wilcoxon(
        pv_diff,
        zero_method="wilcox",
        correction=False,
        alternative="two-sided",
        method="approx",
    )
    pv_rrb = matched_rank_biserial_from_diff(pv_diff)

    pv_diff_boot = pv_diff[boot_idx]
    pv_rrb_boot = vectorized_rank_biserial(pv_diff_boot)
    pv_rrb_ci = percentile_ci(pv_rrb_boot, config.ci_level)

    auxiliary_summary = {
        "n": n,
        "wilcoxon_W": float(pv_w.statistic),
        "wilcoxon_p": float(pv_w.pvalue),
        "median_vertical_minus_planar": float(np.median(pv_diff)),
        "paired_rank_biserial": float(pv_rrb),
        "rank_biserial_CI_low": float(pv_rrb_ci[0]),
        "rank_biserial_CI_high": float(pv_rrb_ci[1]),
    }

    return metric_stats, wasserstein_stats, case_errors, auxiliary_summary


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="both", color=GRID_COLOR, alpha=0.65, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_recovery_metrics(
    metric_stats: pd.DataFrame,
    output_dir: Path,
    config: AnalysisConfig,
) -> None:
    df = metric_stats.set_index("Metric").loc[METRIC_ORDER].reset_index()
    y = np.arange(len(df))
    labels = [
        f"{row.Label} {row.Wilcoxon_Holm_sig}"
        for row in df.itertuples()
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 7.5), sharey=True)
    fig.suptitle(
        f"Engineering-parameter recovery (D1 → D4, n = {int(df['n'].iloc[0])} paired cases)",
        fontsize=20,
        fontweight="bold",
        y=0.98,
    )

    # (a) NMAE
    ax = axes[0]
    vals = df["NMAE_percent"].to_numpy()
    lo = df["NMAE_CI_low_percent"].to_numpy()
    hi = df["NMAE_CI_high_percent"].to_numpy()
    ax.barh(y, vals, color=COLOR_NMAE, edgecolor="none", height=0.58)
    ax.errorbar(vals, y, xerr=np.vstack([vals - lo, hi - vals]), fmt="none",
                ecolor=COLOR_ERROR, elinewidth=1.2, capsize=3)
    for yi, v in zip(y, vals):
        ax.text(v + max(vals) * 0.025, yi, f"{v:.1f}", va="center", fontsize=11, fontweight="bold")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Normalized MAE (%)", fontsize=12)
    ax.set_title("(a) Recovery error\n(lower = better)", fontsize=15, fontweight="bold")
    style_axis(ax)

    # (b) CCC
    ax = axes[1]
    vals = df["CCC"].to_numpy()
    lo = df["CCC_CI_low"].to_numpy()
    hi = df["CCC_CI_high"].to_numpy()
    colors = [COLOR_CCC_GOOD if v >= 0.5 else COLOR_CCC_MID if v >= 0.3 else COLOR_NEUTRAL for v in vals]
    ax.barh(y, vals, color=colors, edgecolor="none", height=0.58)
    ax.errorbar(vals, y, xerr=np.vstack([vals - lo, hi - vals]), fmt="none",
                ecolor=COLOR_ERROR, elinewidth=1.2, capsize=3)
    for yi, v in zip(y, vals):
        ax.text(v + 0.018, yi, f"{v:.2f}", va="center", fontsize=11, fontweight="bold")
    ax.axvline(0, color="#777777", linewidth=0.8)
    ax.set_xlim(min(-0.2, np.nanmin(lo) - 0.05), 1.05)
    ax.set_xlabel("Lin's CCC", fontsize=12)
    ax.set_title("(b) Agreement with original\n(higher = better)", fontsize=15, fontweight="bold")
    style_axis(ax)

    # (c) rank-biserial
    ax = axes[2]
    vals = df["paired_rank_biserial"].to_numpy()
    lo = df["rank_biserial_CI_low"].to_numpy()
    hi = df["rank_biserial_CI_high"].to_numpy()
    sig = df["Wilcoxon_Holm_p"].to_numpy() < 0.05
    colors = [
        COLOR_OVER if (s and v > 0) else COLOR_UNDER if (s and v < 0) else COLOR_NEUTRAL
        for v, s in zip(vals, sig)
    ]
    ax.barh(y, vals, color=colors, edgecolor="none", height=0.58)
    ax.errorbar(vals, y, xerr=np.vstack([vals - lo, hi - vals]), fmt="none",
                ecolor=COLOR_ERROR, elinewidth=1.2, capsize=3)
    ax.axvline(0, color="#777777", linewidth=0.8)
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("Paired rank-biserial r", fontsize=12)
    ax.set_title("(c) Direction of systematic shift\n(D4 − D1)", fontsize=15, fontweight="bold")
    style_axis(ax)
    ax.text(0.23, -0.12, "under-estimate", transform=ax.transAxes, ha="center",
            color=COLOR_UNDER, fontsize=12, fontweight="bold")
    ax.text(0.77, -0.12, "over-estimate", transform=ax.transAxes, ha="center",
            color=COLOR_OVER, fontsize=12, fontweight="bold")

    ccc_legend = [
        Patch(facecolor=COLOR_CCC_GOOD, label="CCC ≥ 0.5"),
        Patch(facecolor=COLOR_CCC_MID, label="0.3–0.5"),
        Patch(facecolor=COLOR_NEUTRAL, label="< 0.3"),
    ]
    shift_legend = [
        Patch(facecolor=COLOR_OVER, label="over (sig.)"),
        Patch(facecolor=COLOR_UNDER, label="under (sig.)"),
        Patch(facecolor=COLOR_NEUTRAL, label="n.s."),
    ]
    fig.legend(handles=ccc_legend, loc="lower center", bbox_to_anchor=(0.56, 0.065),
               ncol=3, frameon=False, fontsize=10)
    fig.legend(handles=shift_legend, loc="lower center", bbox_to_anchor=(0.83, 0.065),
               ncol=3, frameon=False, fontsize=10)
    fig.text(
        0.5, 0.015,
        "NMAE = MAE / IQR(D1). Error bars indicate pair-preserving bootstrap 95% CIs. "
        "Holm-adjusted Wilcoxon: *** p<0.001, ** p<0.01, * p<0.05, n.s. not significant.",
        ha="center", fontsize=10.5, style="italic"
    )
    fig.subplots_adjust(left=0.25, right=0.985, top=0.86, bottom=0.18, wspace=0.18)
    save_figure(fig, output_dir, "Fig_D1_D4_recovery_metrics", config.dpi)


def plot_scatter_grid(
    matched: pd.DataFrame,
    metric_stats: pd.DataFrame,
    output_dir: Path,
    config: AnalysisConfig,
) -> None:
    stat_lookup = metric_stats.set_index("Metric")
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(
        f"Paired case comparison: D1 original vs D4 reconstructed (n = {len(matched)})",
        fontsize=21, fontweight="bold", y=0.985
    )

    for ax, metric in zip(axes.flat, METRIC_ORDER):
        x = matched[f"{metric}_D1"].to_numpy(float)
        y = matched[f"{metric}_D4"].to_numpy(float)
        over = y > x

        ax.scatter(x[over], y[over], s=30, color=COLOR_OVER, alpha=0.72, edgecolors="none")
        ax.scatter(x[~over], y[~over], s=30, color=COLOR_UNDER, alpha=0.78, edgecolors="none")

        combined_min = min(np.min(x), np.min(y))
        combined_max = max(np.max(x), np.max(y))
        span = combined_max - combined_min
        pad = 0.04 * span if span > 0 else 0.1
        lower = combined_min - pad
        upper = combined_max + pad
        if combined_min >= 0 and lower < 0:
            lower = min(0.0, combined_min - 0.01 * span)
        ax.plot([lower, upper], [lower, upper], linestyle="--", color="#777777", linewidth=1.2)
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_aspect("equal", adjustable="box")

        row = stat_lookup.loc[metric]
        ax.text(
            0.035, 0.965,
            f"CCC = {row['CCC']:.2f} [{row['CCC_CI_low']:.2f}, {row['CCC_CI_high']:.2f}]",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#F2F2F2", edgecolor="#AAAAAA", alpha=0.95),
        )
        ax.set_title(METRIC_INFO[metric]["label_unit"], fontsize=14, fontweight="bold")
        ax.set_xlabel("D1 original", fontsize=11)
        ax.set_ylabel("D4 reconstructed", fontsize=11)
        style_axis(ax)

    legend = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=COLOR_OVER,
               markeredgecolor="none", markersize=8, label="D4 > D1 (over-estimate)"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=COLOR_UNDER,
               markeredgecolor="none", markersize=8, label="D4 ≤ D1 (under-estimate)"),
        Line2D([0], [0], linestyle="--", color="#777777", label="Identity line (y = x)"),
    ]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.025),
               ncol=3, frameon=True, fontsize=12)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.94, bottom=0.09, hspace=0.28, wspace=0.22)
    save_figure(fig, output_dir, "Fig_D1_D4_paired_scatter_3x3", config.dpi)


def plot_auxiliary(
    wasserstein_stats: pd.DataFrame,
    case_errors: pd.DataFrame,
    auxiliary_summary: Dict[str, float],
    output_dir: Path,
    config: AnalysisConfig,
) -> None:
    df = wasserstein_stats.set_index("Metric").loc[METRIC_ORDER].reset_index()
    y = np.arange(len(df))

    fig, axes = plt.subplots(1, 2, figsize=(17.5, 8.3), gridspec_kw={"width_ratios": [1.18, 1.0]})
    fig.suptitle(
        "Auxiliary D1–D4 distributional and domain-level error analysis",
        fontsize=20, fontweight="bold", y=0.98
    )

    # (a) normalized Wasserstein distance
    ax = axes[0]
    vals = df["Normalized_W1"].to_numpy()
    lo = df["Normalized_W1_CI_low"].to_numpy()
    hi = df["Normalized_W1_CI_high"].to_numpy()
    sig = df["Permutation_Holm_p"].to_numpy() < 0.05
    colors = [COLOR_PURPLE if s else COLOR_NEUTRAL for s in sig]
    ax.barh(y, vals, color=colors, alpha=0.9, edgecolor="none", height=0.56)
    ax.errorbar(vals, y, xerr=np.vstack([vals - lo, hi - vals]), fmt="none",
                ecolor=COLOR_ERROR, elinewidth=1.2, capsize=3)
    for yi, v in zip(y, vals):
        ax.text(v + max(vals) * 0.022, yi, f"{v:.3f}", va="center", fontsize=11, fontweight="bold")
    ax.set_yticks(y, df["Label"])
    ax.invert_yaxis()
    ax.set_xlabel("Normalized Wasserstein distance  W₁ / IQR(D1)", fontsize=12)
    ax.set_title("(a) Marginal distribution distance", fontsize=15.5, fontweight="bold")
    xmax = max(1.2, np.nanmax(hi) * 1.05)
    ax.set_xlim(0, xmax)
    style_axis(ax)

    # (b) predefined planar-vertical comparison
    ax = axes[1]
    xp = case_errors["E_planar"].to_numpy(float)
    yv = case_errors["E_vertical"].to_numpy(float)
    over = yv > xp
    ax.scatter(xp[over], yv[over], s=42, color=COLOR_OVER, alpha=0.85, edgecolors="none")
    ax.scatter(xp[~over], yv[~over], s=42, color=COLOR_UNDER, alpha=0.85, edgecolors="none")
    upper = max(np.max(xp), np.max(yv)) * 1.05
    ax.plot([0, upper], [0, upper], linestyle="--", color="#777777", linewidth=1.2)
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_xlabel("Planar mean standardized error", fontsize=12)
    ax.set_ylabel("Vertical mean standardized error", fontsize=12)
    ax.set_title("(b) Predefined planar–vertical comparison", fontsize=15.5, fontweight="bold")
    style_axis(ax)

    text = (
        f"Wilcoxon p = {format_p(auxiliary_summary['wilcoxon_p'])}\n"
        f"r_rb = {auxiliary_summary['paired_rank_biserial']:.3f} "
        f"[{auxiliary_summary['rank_biserial_CI_low']:.3f}, "
        f"{auxiliary_summary['rank_biserial_CI_high']:.3f}]\n"
        f"median(V−P) = {auxiliary_summary['median_vertical_minus_planar']:.3f}"
    )
    ax.text(
        0.04, 0.96, text, transform=ax.transAxes, va="top", fontsize=11.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#AAAAAA", alpha=0.95)
    )
    aux_legend = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=COLOR_OVER,
               markeredgecolor="none", markersize=8, label="Vertical > planar"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=COLOR_UNDER,
               markeredgecolor="none", markersize=8, label="Vertical ≤ planar"),
        Line2D([0], [0], linestyle="--", color="#777777", label="Equality line"),
    ]
    ax.legend(handles=aux_legend, loc="lower right", frameon=False, fontsize=11)

    w_legend = [
        Patch(facecolor=COLOR_PURPLE, label="difference detected (Holm-adjusted p < 0.05)"),
        Patch(facecolor=COLOR_NEUTRAL, label="no clear evidence of difference"),
    ]
    fig.legend(handles=w_legend, loc="lower left", bbox_to_anchor=(0.07, 0.07),
               ncol=2, frameon=False, fontsize=10.5)
    fig.text(
        0.5, 0.025,
        f"Error bars are pair-preserving bootstrap 95% CIs ({config.n_bootstrap:,} resamples). "
        "FAR was predefined as a bridge/hybrid metric and excluded from the planar–vertical contrast.",
        ha="center", fontsize=10.5, style="italic"
    )
    fig.subplots_adjust(left=0.17, right=0.985, top=0.88, bottom=0.16, wspace=0.22)
    save_figure(fig, output_dir, "Fig_D1_D4_auxiliary_distribution_planar_vertical", config.dpi)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate D1-D4 morphological recovery statistics and figures."
    )
    parser.add_argument("--input-xlsx", type=Path, required=True, help="Workbook containing D1 and D4 sheets.")
    parser.add_argument("--d1-sheet", default="D1_internal_100")
    parser.add_argument("--d4-sheet", default="D4_internal_100")
    parser.add_argument("--case-id-col", default="case_id")
    parser.add_argument("--output-dir", type=Path, default=Path("d1_d4_outputs"))
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--n-permutations", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = AnalysisConfig(
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        random_seed=args.random_seed,
        dpi=args.dpi,
    )

    if config.n_bootstrap < 100:
        raise ValueError("--n-bootstrap should be at least 100.")
    if config.n_permutations < 100:
        raise ValueError("--n-permutations should be at least 100.")
    if not args.input_xlsx.exists():
        raise FileNotFoundError(args.input_xlsx)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    matched = load_and_match(
        args.input_xlsx,
        args.d1_sheet,
        args.d4_sheet,
        args.case_id_col,
    )
    check_tpi_consistency(matched, args.case_id_col, config.tpi_tolerance, args.output_dir)

    metric_stats, wasserstein_stats, case_errors, auxiliary_summary = analyze_metrics(
        matched, args.case_id_col, config
    )

    metric_stats.to_csv(args.output_dir / "D1_D4_statistics.csv", index=False)
    wasserstein_stats.to_csv(args.output_dir / "Wasserstein_results.csv", index=False)
    case_errors.to_csv(args.output_dir / "Planar_vertical_case_level_errors.csv", index=False)

    with open(args.output_dir / "Planar_vertical_summary.json", "w", encoding="utf-8") as f:
        json.dump(auxiliary_summary, f, indent=2, ensure_ascii=False)

    run_metadata = {
        "input_xlsx": str(args.input_xlsx),
        "d1_sheet": args.d1_sheet,
        "d4_sheet": args.d4_sheet,
        "case_id_column": args.case_id_col,
        "n_matched_cases": len(matched),
        "n_bootstrap": config.n_bootstrap,
        "n_permutations": config.n_permutations,
        "random_seed_bootstrap": config.random_seed,
        "random_seed_permutation": config.random_seed + 1,
        "ci_level": config.ci_level,
        "metric_order": METRIC_ORDER,
        "vertical_metrics": VERTICAL_METRICS,
        "planar_metrics": PLANAR_METRICS,
        "far_role": "hybrid/bridge; excluded from planar-vertical contrast",
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    with open(args.output_dir / "analysis_run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2, ensure_ascii=False)

    plot_recovery_metrics(metric_stats, args.output_dir, config)
    plot_scatter_grid(matched, metric_stats, args.output_dir, config)
    plot_auxiliary(wasserstein_stats, case_errors, auxiliary_summary, args.output_dir, config)

    print(f"Completed analysis for {len(matched)} matched cases.")
    print(f"Outputs written to: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
