"""Create a journal-style per-ROI two-channel colocalization figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


CONDITION_COLORS = ("#69B3D0", "#E77AA8")
METRIC = "pearson_r_background_corrected_whole_roi"


def significance_label(p_value: float) -> str:
    if p_value < 0.0001:
        return "****"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def load_values(
    source: Path, condition_order: tuple[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], tuple[str, str]]:
    data = pd.read_csv(source)
    required = {"condition", "acquisition_date", "replicate", METRIC}
    if missing := required.difference(data.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    data["condition"] = data["condition"].astype(str).str.strip().str.casefold()
    observed = tuple(sorted(data["condition"].dropna().unique()))
    if condition_order is None:
        if len(observed) != 2:
            raise ValueError(
                f"Exactly two conditions are required or specify --condition-a/--condition-b; found {observed}"
            )
        condition_order = observed
    if missing_conditions := sorted(set(condition_order) - set(observed)):
        raise ValueError(f"Requested conditions not found: {missing_conditions}")
    data[METRIC] = pd.to_numeric(data[METRIC], errors="raise")
    if data[METRIC].isna().any():
        raise ValueError("Primary Pearson coefficient contains missing values")
    values = {
        condition: data.loc[data["condition"] == condition, METRIC].to_numpy(float)
        for condition in condition_order
    }
    return data, values, condition_order


def calculate_statistics(
    values: dict[str, np.ndarray], condition_order: tuple[str, str]
) -> dict:
    group_a, group_b = (values[name] for name in condition_order)
    test = stats.ttest_ind(group_a, group_b, equal_var=True)
    return {
        "metric": METRIC,
        "statistical_unit": "individual nucleus",
        "test": "unpaired two-sided t-test; equal variance",
        "condition_a": condition_order[0],
        "condition_b": condition_order[1],
        "group_a_n": len(group_a),
        "group_b_n": len(group_b),
        "group_a_mean": float(np.mean(group_a)),
        "group_a_sem": float(stats.sem(group_a, ddof=1)),
        "group_b_mean": float(np.mean(group_b)),
        "group_b_sem": float(stats.sem(group_b, ddof=1)),
        "t_statistic": float(test.statistic),
        "degrees_of_freedom": len(group_a) + len(group_b) - 2,
        "p_value": float(test.pvalue),
        "significance": significance_label(float(test.pvalue)),
    }


def plot(
    values: dict[str, np.ndarray], result: dict, output_stem: Path, ylabel: str,
    condition_order: tuple[str, str], condition_labels: tuple[str, str],
) -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 10,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    panel_inches = 5 / 2.54
    figure, axis = plt.subplots(
        figsize=(panel_inches, panel_inches), constrained_layout=True
    )
    positions = np.arange(2, dtype=float)
    means = np.array([result["group_a_mean"], result["group_b_mean"]])
    sems = np.array([result["group_a_sem"], result["group_b_sem"]])
    axis.bar(
        positions, means, width=0.52, color=CONDITION_COLORS,
        edgecolor="none", alpha=0.82, zorder=1,
    )
    axis.errorbar(
        positions, means, yerr=sems, fmt="none", ecolor="#1A1A1A",
        elinewidth=0.5, capsize=3, capthick=0.5, zorder=5,
    )
    rng = np.random.default_rng(20260816)
    for position, condition in enumerate(condition_order):
        observed = values[condition]
        jitter = rng.uniform(-0.15, 0.15, size=len(observed))
        axis.scatter(
            np.full(len(observed), position) + jitter, observed,
            s=1, color="#111111", alpha=1.0, linewidths=0,
            zorder=2, rasterized=True,
        )

    significance_y = min(0.985, max(0.90, float(max(np.max(x) for x in values.values())) + 0.015))
    axis.plot([0, 1], [significance_y, significance_y], color="#444444", linewidth=0.5)
    axis.text(
        0.5, significance_y + 0.012, result["significance"],
        ha="center", va="bottom", fontsize=10,
        fontweight="bold" if result["significance"] != "ns" else "normal",
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(condition_labels, fontsize=10, fontweight="bold")
    for position, count in zip(positions, (result["group_a_n"], result["group_b_n"])):
        axis.text(
            position, -0.135, f"n={count}", transform=axis.get_xaxis_transform(),
            ha="center", va="top", fontsize=10, fontweight="normal", clip_on=False,
        )
    axis.set_xlim(-0.30, 1.30)
    axis.set_ylim(0, 1.06)
    axis.set_yticks(np.arange(0, 1.01, 0.2))
    axis.set_ylabel(ylabel, fontsize=10, fontweight="bold", multialignment="center")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(1.0)
    axis.spines["bottom"].set_linewidth(1.0)
    axis.tick_params(axis="both", direction="out", width=1.0, length=3, labelsize=10)
    plt.setp(axis.get_yticklabels(), fontsize=10, fontweight="bold")
    axis.grid(False)

    figure.savefig(output_stem.with_suffix(".png"), dpi=600, facecolor="white")
    figure.savefig(output_stem.with_suffix(".pdf"), facecolor="white")
    figure.savefig(output_stem.with_suffix(".svg"), facecolor="white")
    plt.close(figure)


def run(
    source: Path, output_dir: Path, ylabel: str | None = None,
    condition_order: tuple[str, str] | None = None,
    condition_labels: tuple[str, str] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data, values, condition_order = load_values(source, condition_order)
    result = calculate_statistics(values, condition_order)
    if ylabel is None:
        target_a = str(data["target_a"].dropna().iloc[0]) if "target_a" in data else "Target A"
        target_b = str(data["target_b"].dropna().iloc[0]) if "target_b" in data else "Target B"
        ylabel = f"{target_a}–{target_b}\nPearson r"
    stem = output_dir / "two_channel_Pearson_colocalization"
    condition_labels = condition_labels or tuple(name.title() for name in condition_order)
    plot(values, result, stem, ylabel, condition_order, condition_labels)
    pd.DataFrame([result]).to_csv(output_dir / "colocalization_statistics.csv", index=False)
    (
        data.groupby(["acquisition_date", "replicate", "condition"], as_index=False)
        .agg(mean=(METRIC, "mean"), sem=(METRIC, "sem"), nuclei=(METRIC, "size"))
        .to_csv(output_dir / "colocalization_replicate_QC.csv", index=False)
    )
    print(pd.DataFrame([result]).to_string(index=False))


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path,
        default=root / "two_channel_colocalization_results" / "per_roi_colocalization.csv",
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--ylabel", help="Optional y-axis label; use \\n for a line break")
    parser.add_argument("--condition-a", help="First condition value; auto-detected if omitted")
    parser.add_argument("--condition-b", help="Second condition value; auto-detected if omitted")
    parser.add_argument("--condition-a-label", help="Displayed label for the first condition")
    parser.add_argument("--condition-b-label", help="Displayed label for the second condition")
    args = parser.parse_args()
    if bool(args.condition_a) != bool(args.condition_b):
        raise SystemExit("Provide both --condition-a and --condition-b, or neither")
    order = (args.condition_a.casefold(), args.condition_b.casefold()) if args.condition_a else None
    labels = None
    if args.condition_a_label or args.condition_b_label:
        if not (args.condition_a_label and args.condition_b_label):
            raise SystemExit("Provide both display labels, or neither")
        labels = (args.condition_a_label, args.condition_b_label)
    run(args.source.resolve(), args.output.resolve(), args.ylabel, order, labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
