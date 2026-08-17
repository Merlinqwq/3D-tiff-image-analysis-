"""Create journal-style focus-volume and foci-per-nucleus figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


CONDITION_ORDER = ("normoxia", "hypoxia")
CONDITION_LABELS = ("Normoxia", "Hypoxia")
CONDITION_COLORS = ("#69B3D0", "#E77AA8")
TARGET_DISPLAY = {"coilin": "Coilin"}


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


def target_display(target: str) -> str:
    return TARGET_DISPLAY.get(target.casefold(), target)


def load_data(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    nuclei = pd.read_csv(root / "Merged_condensate_nucleus_focus_counts.csv")
    foci = pd.read_csv(root / "Merged_condensate_focus_volumes.csv")
    nucleus_required = {
        "acquisition_date", "condition", "replicate", "target", "focus_count",
    }
    focus_required = {
        "acquisition_date", "condition", "replicate", "target", "volume_um3",
    }
    if missing := nucleus_required.difference(nuclei.columns):
        raise ValueError(f"Missing nucleus columns: {sorted(missing)}")
    if missing := focus_required.difference(foci.columns):
        raise ValueError(f"Missing focus columns: {sorted(missing)}")
    for frame in (nuclei, foci):
        frame["condition"] = frame["condition"].astype(str).str.strip().str.casefold()
        unexpected = sorted(set(frame["condition"]) - set(CONDITION_ORDER))
        if unexpected:
            raise ValueError(f"Unexpected conditions: {unexpected}")
        frame["acquisition_date"] = frame["acquisition_date"].astype(str)
    nuclei["focus_count"] = pd.to_numeric(nuclei["focus_count"], errors="raise")
    foci["volume_um3"] = pd.to_numeric(foci["volume_um3"], errors="raise")
    if nuclei["focus_count"].isna().any() or foci["volume_um3"].isna().any():
        raise ValueError("Missing focus counts or focus volumes")
    return nuclei, foci


def calculate_one(
    target: str,
    metric_name: str,
    values_by_condition: dict[str, np.ndarray],
    statistical_unit: str,
) -> dict:
    normoxia = values_by_condition["normoxia"]
    hypoxia = values_by_condition["hypoxia"]
    if len(normoxia) < 2 or len(hypoxia) < 2:
        raise ValueError(f"At least two observations per condition required: {target} {metric_name}")
    test = stats.ttest_ind(normoxia, hypoxia, equal_var=True)
    return {
        "target": target,
        "metric": metric_name,
        "statistical_unit": statistical_unit,
        "test": "unpaired two-sided t-test; equal variance",
        "normoxia_n": len(normoxia),
        "hypoxia_n": len(hypoxia),
        "normoxia_mean": float(np.mean(normoxia)),
        "normoxia_sem": float(stats.sem(normoxia, ddof=1)),
        "hypoxia_mean": float(np.mean(hypoxia)),
        "hypoxia_sem": float(stats.sem(hypoxia, ddof=1)),
        "t_statistic": float(test.statistic),
        "degrees_of_freedom": len(normoxia) + len(hypoxia) - 2,
        "p_value": float(test.pvalue),
        "significance": significance_label(float(test.pvalue)),
    }


def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(1.0)
    axis.spines["bottom"].set_linewidth(1.0)
    axis.tick_params(axis="both", direction="out", width=1.0, length=3, labelsize=10)
    axis.grid(False)
    plt.setp(axis.get_yticklabels(), fontsize=10, fontweight="bold")


def plot_metric(
    axis: plt.Axes,
    values_by_condition: dict[str, np.ndarray],
    statistics: dict,
    ylabel: str,
    rng: np.random.Generator,
) -> None:
    x_positions = np.arange(2, dtype=float)
    means = np.array(
        [statistics["normoxia_mean"], statistics["hypoxia_mean"]], dtype=float
    )
    sems = np.array(
        [statistics["normoxia_sem"], statistics["hypoxia_sem"]], dtype=float
    )
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
        values = values_by_condition[condition]
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

    maximum = float(
        max(max(np.max(values) for values in values_by_condition.values()), np.max(means + sems))
    )
    significance_y = maximum * 1.09 if maximum > 0 else 1.0
    axis.plot(
        [0, 1], [significance_y, significance_y],
        color="#444444", linewidth=0.5, clip_on=False,
    )
    axis.text(
        0.5,
        significance_y + maximum * 0.018,
        statistics["significance"],
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold" if statistics["significance"] != "ns" else "normal",
    )
    axis.set_xticks(x_positions)
    axis.set_xticklabels(CONDITION_LABELS, fontsize=10, fontweight="bold")
    for x_position, count in zip(
        x_positions, (statistics["normoxia_n"], statistics["hypoxia_n"])
    ):
        axis.text(
            x_position,
            -0.135,
            f"n={count}",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=10,
            fontweight="normal",
            clip_on=False,
        )
    axis.set_xlim(-0.30, 1.30)
    axis.set_ylim(0, significance_y + maximum * 0.13 if maximum > 0 else 1.2)
    axis.set_ylabel(ylabel, fontsize=10, fontweight="bold", multialignment="center")
    style_axis(axis)


def save_figure(figure: plt.Figure, path_without_suffix: Path) -> None:
    figure.savefig(path_without_suffix.with_suffix(".png"), dpi=600, facecolor="white")
    figure.savefig(path_without_suffix.with_suffix(".pdf"), facecolor="white")
    figure.savefig(path_without_suffix.with_suffix(".svg"), facecolor="white")
    plt.close(figure)


def analyze(root: Path, output_dir: Path) -> None:
    root = root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    nuclei, foci = load_data(root)
    targets = sorted(set(nuclei["target"]) & set(foci["target"]), key=str.casefold)
    if not targets:
        raise ValueError("No shared targets found in nucleus and individual-focus data")

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
    panel_inches = 5 / 2.54
    statistics_rows: list[dict] = []
    replicate_rows: list[dict] = []

    for target_index, target in enumerate(targets):
        target_folder = output_dir / str(target)
        target_folder.mkdir(exist_ok=True)
        display = target_display(str(target))
        nucleus_subset = nuclei[nuclei["target"] == target]
        focus_subset = foci[foci["target"] == target]
        metric_specs = [
            (
                "focus_volume_um3",
                {
                    condition: focus_subset.loc[
                        focus_subset["condition"] == condition, "volume_um3"
                    ].to_numpy(dtype=float)
                    for condition in CONDITION_ORDER
                },
                "individual focus",
                f"{display} focus volume\n(µm$^3$)",
            ),
            (
                "foci_number_per_nucleus",
                {
                    condition: nucleus_subset.loc[
                        nucleus_subset["condition"] == condition, "focus_count"
                    ].to_numpy(dtype=float)
                    for condition in CONDITION_ORDER
                },
                "individual nucleus",
                f"{display} foci number\nper nucleus",
            ),
        ]
        target_statistics = []
        for metric_index, (metric, values, unit, ylabel) in enumerate(metric_specs):
            result = calculate_one(str(target), metric, values, unit)
            statistics_rows.append(result)
            target_statistics.append((metric, values, result, ylabel))
            figure, axis = plt.subplots(
                figsize=(panel_inches, panel_inches), constrained_layout=True
            )
            plot_metric(
                axis,
                values,
                result,
                ylabel,
                np.random.default_rng(20260816 + target_index * 10 + metric_index),
            )
            save_figure(figure, target_folder / f"{target}_{metric}")

            source = focus_subset if metric == "focus_volume_um3" else nucleus_subset
            value_column = "volume_um3" if metric == "focus_volume_um3" else "focus_count"
            for row in (
                source.groupby(["acquisition_date", "replicate", "condition"], as_index=False)
                .agg(mean=(value_column, "mean"), observations=(value_column, "size"))
                .itertuples(index=False)
            ):
                replicate_rows.append(
                    {
                        "target": target,
                        "metric": metric,
                        "acquisition_date": row.acquisition_date,
                        "replicate": row.replicate,
                        "condition": row.condition,
                        "mean": float(row.mean),
                        "observations": int(row.observations),
                    }
                )

        figure, axes = plt.subplots(
            1,
            2,
            figsize=((5 * 2 + 2.5) / 2.54, panel_inches),
            constrained_layout=True,
        )
        for metric_index, (axis, (_, values, result, ylabel)) in enumerate(
            zip(axes, target_statistics)
        ):
            plot_metric(
                axis,
                values,
                result,
                ylabel,
                np.random.default_rng(20260816 + target_index * 10 + metric_index),
            )
        save_figure(figure, target_folder / f"{target}_focus_volume_and_number")

    statistics = pd.DataFrame(statistics_rows)
    replicate_qc = pd.DataFrame(replicate_rows)
    statistics.to_csv(output_dir / "condensate_foci_statistics.csv", index=False)
    replicate_qc.to_csv(output_dir / "condensate_foci_replicate_means_QC.csv", index=False)
    print(f"TARGETS={len(targets)}")
    print(f"FIGURE_SETS={len(targets) * 3}")
    print(statistics[["target", "metric", "normoxia_n", "hypoxia_n", "p_value", "significance"]].to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    analyze(args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
