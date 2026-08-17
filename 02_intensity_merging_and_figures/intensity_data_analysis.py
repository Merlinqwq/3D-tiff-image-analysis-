"""Shared analysis and journal-style plotting for merged nuclear intensity data."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


CONDITION_ORDER = ("normoxia", "hypoxia")
CONDITION_LABELS = ("Normoxia", "Hypoxia")
CONDITION_COLORS = ("#69B3D0", "#E77AA8")


def significance_label(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "NA"
    if p_value < 0.0001:
        return "****"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def display_name(metric: str) -> str:
    name = re.sub(r"_integrated_intensity$", "", metric, flags=re.IGNORECASE)
    return "DAPI" if name.casefold() == "dapi" else name


def scale_for_values(values: np.ndarray) -> tuple[float, int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    maximum = float(np.max(np.abs(finite))) if finite.size else 0.0
    if maximum < 1_000:
        return 1.0, 0
    exponent = int(math.floor(math.log10(maximum) / 3) * 3)
    return float(10**exponent), exponent


def load_and_validate(workbook_path: Path) -> tuple[pd.DataFrame, list[str]]:
    data = pd.read_excel(workbook_path, sheet_name="Nuclei")
    required = {"acquisition_date", "condition"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns in {workbook_path.name}: {sorted(missing)}")

    metrics = [column for column in data.columns if column.endswith("_integrated_intensity")]
    if not metrics:
        raise ValueError(f"No integrated-intensity columns found in {workbook_path.name}")

    data = data.copy()
    data["condition"] = data["condition"].astype(str).str.strip().str.casefold()
    unexpected = sorted(set(data["condition"]) - set(CONDITION_ORDER))
    if unexpected:
        raise ValueError(f"Unexpected conditions in {workbook_path.name}: {unexpected}")
    data["acquisition_date"] = pd.to_datetime(data["acquisition_date"]).dt.strftime("%Y-%m-%d")
    for metric in metrics:
        data[metric] = pd.to_numeric(data[metric], errors="raise")
        if data[metric].isna().any():
            raise ValueError(f"Missing values detected in {metric}")
    return data, metrics


def calculate_statistics(
    data: pd.DataFrame,
    metrics: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict] = []
    replicate_rows: list[dict] = []

    for metric in metrics:
        grouped = (
            data.groupby(["acquisition_date", "condition"], as_index=False)[metric]
            .mean()
            .rename(columns={metric: "replicate_mean"})
        )
        norm_values = data.loc[data["condition"] == "normoxia", metric].to_numpy(dtype=float)
        hyp_values = data.loc[data["condition"] == "hypoxia", metric].to_numpy(dtype=float)
        if len(norm_values) < 2 or len(hyp_values) < 2:
            raise ValueError(f"At least two nuclei per condition are required for {metric}")

        test = stats.ttest_ind(norm_values, hyp_values, equal_var=True)
        norm_mean = float(np.mean(norm_values))
        hyp_mean = float(np.mean(hyp_values))
        norm_sem = float(stats.sem(norm_values, ddof=1))
        hyp_sem = float(stats.sem(hyp_values, ddof=1))
        p_value = float(test.pvalue)

        summary_rows.append(
            {
                "metric": metric,
                "display_name": display_name(metric),
                "statistical_unit": "individual nucleus",
                "test": "unpaired two-sided t-test; equal variance",
                "normoxia_replicates": int(
                    data.loc[data["condition"] == "normoxia", "acquisition_date"].nunique()
                ),
                "hypoxia_replicates": int(
                    data.loc[data["condition"] == "hypoxia", "acquisition_date"].nunique()
                ),
                "normoxia_nuclei": int(len(norm_values)),
                "hypoxia_nuclei": int(len(hyp_values)),
                "normoxia_mean": norm_mean,
                "normoxia_sem": norm_sem,
                "hypoxia_mean": hyp_mean,
                "hypoxia_sem": hyp_sem,
                "hypoxia_over_normoxia": hyp_mean / norm_mean if norm_mean else np.nan,
                "difference_hypoxia_minus_normoxia": hyp_mean - norm_mean,
                "t_statistic": float(test.statistic),
                "degrees_of_freedom": int(len(norm_values) + len(hyp_values) - 2),
                "p_value": p_value,
                "significance": significance_label(p_value),
            }
        )

        for row in grouped.itertuples(index=False):
            date = row.acquisition_date
            condition = row.condition
            if condition in CONDITION_ORDER:
                replicate_rows.append(
                    {
                        "metric": metric,
                        "acquisition_date": date,
                        "condition": condition,
                        "replicate_mean": float(row.replicate_mean),
                        "nuclei_in_replicate": int(
                            ((data["acquisition_date"] == date) & (data["condition"] == condition)).sum()
                        ),
                    }
                )

    return pd.DataFrame(summary_rows), pd.DataFrame(replicate_rows)


def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(1.0)
    axis.spines["bottom"].set_linewidth(1.0)
    axis.tick_params(axis="both", direction="out", width=1.0, length=3, labelsize=10)
    axis.grid(False)


def plot_metric(
    axis: plt.Axes,
    data: pd.DataFrame,
    metric: str,
    summary_row: pd.Series,
    _replicate_means: pd.DataFrame,
    rng: np.random.Generator,
) -> None:
    raw_values = data[metric].to_numpy(dtype=float)
    scale, exponent = scale_for_values(raw_values)
    scaled = data.assign(_plot_value=data[metric] / scale)
    means = np.array(
        [summary_row["normoxia_mean"], summary_row["hypoxia_mean"]], dtype=float
    ) / scale
    sems = np.array(
        [summary_row["normoxia_sem"], summary_row["hypoxia_sem"]], dtype=float
    ) / scale
    x_positions = np.arange(2, dtype=float)

    axis.bar(
        x_positions,
        means,
        width=0.52,
        color=CONDITION_COLORS,
        edgecolor="none",
        alpha=0.82,
        zorder=1,
    )
    axis.errorbar(
        x_positions,
        means,
        yerr=sems,
        fmt="none",
        ecolor="#1A1A1A",
        elinewidth=0.5,
        capsize=3,
        capthick=0.5,
        zorder=5,
    )

    for x_index, condition in enumerate(CONDITION_ORDER):
        values = scaled.loc[scaled["condition"] == condition, "_plot_value"].to_numpy(dtype=float)
        jitter = rng.uniform(-0.15, 0.15, size=len(values))
        axis.scatter(
            np.full(len(values), x_positions[x_index]) + jitter,
            values,
            s=1,
            color="#111111",
            alpha=1.0,
            linewidths=0,
            zorder=2,
            rasterized=True,
        )

    maximum = float(max(scaled["_plot_value"].max(), np.max(means + sems)))
    significance_y = maximum * 1.09 if maximum > 0 else 1.0
    axis.plot(
        [0, 1],
        [significance_y, significance_y],
        color="#444444",
        linewidth=0.5,
        clip_on=False,
    )
    axis.text(
        0.5,
        significance_y + maximum * 0.018,
        summary_row["significance"],
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold" if summary_row["significance"] != "ns" else "normal",
    )

    n_norm = int(summary_row["normoxia_nuclei"])
    n_hyp = int(summary_row["hypoxia_nuclei"])
    axis.set_xticks(x_positions)
    axis.set_xticklabels(
        CONDITION_LABELS,
        fontsize=10,
        fontweight="bold",
    )
    for x_position, nucleus_count in zip(x_positions, (n_norm, n_hyp)):
        axis.text(
            x_position,
            -0.135,
            f"n={nucleus_count}",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=10,
            fontweight="normal",
            clip_on=False,
        )
    axis.set_xlim(-0.30, 1.30)
    axis.set_ylim(0, significance_y + maximum * 0.085 if maximum > 0 else 1.2)
    target = display_name(metric)
    unit = f"{target} integrated\nintensity (a.u.)"
    if exponent != 0:
        unit = f"{target} integrated\nintensity (×10$^{exponent}$ a.u.)"
    axis.set_ylabel(unit, fontsize=10, fontweight="bold", multialignment="center")
    style_axis(axis)
    plt.setp(axis.get_yticklabels(), fontsize=10, fontweight="bold")


def save_single_metric_figure(
    data: pd.DataFrame,
    metric: str,
    summary_row: pd.Series,
    replicate_means: pd.DataFrame,
    output_dir: Path,
    analysis_name: str,
    seed: int,
) -> None:
    panel_inches = 5 / 2.54
    figure, axis = plt.subplots(figsize=(panel_inches, panel_inches), constrained_layout=True)
    plot_metric(axis, data, metric, summary_row, replicate_means, np.random.default_rng(seed))
    stem = f"{analysis_name}_{metric}"
    figure.savefig(output_dir / f"{stem}.png", dpi=600, facecolor="white")
    figure.savefig(output_dir / f"{stem}.pdf", facecolor="white")
    figure.savefig(output_dir / f"{stem}.svg", facecolor="white")
    plt.close(figure)


def analyze_workbook(workbook_path: Path, output_dir: Path, analysis_name: str) -> None:
    workbook_path = workbook_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data, all_metrics = load_and_validate(workbook_path)
    metrics = [metric for metric in all_metrics if display_name(metric).casefold() != "dapi"]
    if not metrics:
        raise ValueError(f"No target-protein integrated-intensity columns found in {workbook_path.name}")
    summary, replicate_means = calculate_statistics(data, metrics)

    for suffix in ("png", "pdf", "svg"):
        stale_dapi = output_dir / f"{analysis_name}_DAPI_integrated_intensity.{suffix}"
        if stale_dapi.exists():
            stale_dapi.unlink()

    summary.to_csv(output_dir / f"{analysis_name}_statistics.csv", index=False)
    replicate_means.to_csv(output_dir / f"{analysis_name}_replicate_means.csv", index=False)

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    for metric_index, metric in enumerate(metrics):
        summary_row = summary.loc[summary["metric"] == metric].iloc[0]
        save_single_metric_figure(
            data,
            metric,
            summary_row,
            replicate_means,
            output_dir,
            analysis_name,
            seed=20260816 + metric_index,
        )

    panel_inches = 5 / 2.54
    figure, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(panel_inches * len(metrics), panel_inches),
        squeeze=False,
        constrained_layout=True,
    )
    for metric_index, (axis, metric) in enumerate(zip(axes[0], metrics)):
        summary_row = summary.loc[summary["metric"] == metric].iloc[0]
        plot_metric(
            axis,
            data,
            metric,
            summary_row,
            replicate_means,
            np.random.default_rng(20260816 + metric_index),
        )
    figure.savefig(
        output_dir / f"{analysis_name}_all_metrics.png",
        dpi=600,
        facecolor="white",
    )
    figure.savefig(
        output_dir / f"{analysis_name}_all_metrics.pdf",
        facecolor="white",
    )
    figure.savefig(
        output_dir / f"{analysis_name}_all_metrics.svg",
        facecolor="white",
    )
    plt.close(figure)

    print(f"WORKBOOK={workbook_path}")
    print(f"OUTPUT={output_dir}")
    print(
        summary[
            ["display_name", "normoxia_nuclei", "hypoxia_nuclei", "p_value", "significance"]
        ].to_string(index=False)
    )


def cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("analysis_name")
    args = parser.parse_args()
    analyze_workbook(args.workbook, args.output_dir, args.analysis_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
