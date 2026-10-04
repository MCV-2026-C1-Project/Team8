"""Generate publication-ready figures for Week 1 image retrieval results (CIELab & HSV).

Outputs saved to plots/:
  - lab_distances_comparison.png : Grouped bar plot (mAP@1 & mAP@5) across measures for Lab 8-64-64.
  - lab_top10_bins.png           : Grouped bar plot of top-10 CIELab setups.
  - lab_lightness_effect.png     : Grouped bar plot showing Lightness (L) suppression effect.
  - lab_measures_heatmap.png     : Heatmap of (bins x measures) across mAP@5 for CIELab.
  - hsv_distances_comparison.png : Grouped bar plot across measures for top HSV bins.
  - hsv_top10_configs.png        : Grouped bar plot of top-10 HSV configurations.
  - hsv_parameters_effect.png    : Grouped bar plot showing effect of HSV weighting & smoothing.
  - hsv_measures_heatmap.png     : Heatmap of (bins x measures) across mAP@5 for HSV.
"""
from pathlib import Path
import ast
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config_results"
PLOTS_DIR = ROOT / "plots"


def setup_style():
    """Apply consistent styling for scientific reporting."""
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams.update({
        "figure.autolayout": True,
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.edgecolor": "#cccccc",
        "axes.linewidth": 0.8,
    })


def annotate_bars(ax, fmt="{:.2f}", fontsize=8.5, offset=2):
    """Add numerical score labels on top of bar charts."""
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(
                fmt.format(h),
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                fontsize=fontsize,
                xytext=(0, offset),
                textcoords="offset points",
            )


# -----------------------------------------------------------------------------
# Common Plots: Distances & Top-10
# -----------------------------------------------------------------------------
def plot_distances_for_descriptor(df, out_dir, desc_name, target_bins):
    """Grouped bar plot comparing all distance measures for a fixed bin setup."""
    subset = df[(df["descriptor"] == desc_name) & (df["bins"] == target_bins)].copy()
    if subset.empty:
        subset = df[df["descriptor"] == desc_name].copy()
        if subset.empty:
            return
        target_bins = subset.sort_values(by="map5", ascending=False)["bins"].iloc[0]
        subset = subset[subset["bins"] == target_bins]

    # If multiple parameter runs exist for the same measure, take the best
    subset = subset.sort_values(by="map5", ascending=False).groupby("measure").first().reset_index()
    subset = subset.sort_values(by="map5", ascending=False)

    melted = pd.melt(
        subset,
        id_vars=["measure"],
        value_vars=["map1", "map5"],
        var_name="Metric",
        value_name="Score",
    )
    melted["Metric"] = melted["Metric"].replace({"map1": "mAP@1", "map5": "mAP@5"})

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(
        data=melted,
        x="measure",
        y="Score",
        hue="Metric",
        palette=["#4C72B0", "#55A868"],
        edgecolor="black",
        linewidth=0.6,
        ax=ax,
    )

    ax.set_title(
        f"Similarity Measures Comparison for {desc_name.upper()} ({target_bins})",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Distance / Similarity Measure", fontweight="bold")
    ax.set_ylabel("Mean Average Precision", fontweight="bold")
    ax.set_ylim(0, max(0.75, subset["map5"].max() + 0.1))
    ax.legend(title="Metric", loc="upper right")
    annotate_bars(ax)

    out_path = out_dir / f"{desc_name}_distances_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_top10_configurations(df, out_dir, desc_name):
    """Grouped bar plot showing both mAP@1 and mAP@5 for the top 10 setups."""
    subset = df[df["descriptor"] == desc_name].copy()
    if subset.empty:
        return

    subset["config_label"] = subset["bins"] + "\n(" + subset["measure"] + ")"
    top10 = subset.sort_values(by="map5", ascending=False).head(10).copy()

    melted = pd.melt(
        top10,
        id_vars=["config_label"],
        value_vars=["map1", "map5"],
        var_name="Metric",
        value_name="Score",
    )
    melted["Metric"] = melted["Metric"].replace({"map1": "mAP@1", "map5": "mAP@5"})

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=melted,
        x="config_label",
        y="Score",
        hue="Metric",
        palette=["#4C72B0", "#55A868"],
        edgecolor="black",
        linewidth=0.6,
        ax=ax,
    )

    ax.set_title(
        f"Top-10 {desc_name.upper()} Configurations Ranked by mAP@5",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Configuration: Bins (Measure)", fontweight="bold")
    ax.set_ylabel("Mean Average Precision", fontweight="bold")
    ax.set_ylim(0, max(0.75, top10["map5"].max() + 0.1))
    ax.legend(title="Metric", loc="upper right")
    annotate_bars(ax)

    out_path = out_dir / f"{desc_name}_top10_bins.png" if desc_name == "lab" else out_dir / f"{desc_name}_top10_configs.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_measures_heatmap(df, out_dir, desc_name):
    """Heatmap showing mAP@5 across bin configurations and distance measures."""
    subset = df[df["descriptor"] == desc_name].copy()
    if subset.empty:
        return

    pivot_m5 = subset.groupby(["bins", "measure"])["map5"].max().unstack()
    row_order = pivot_m5.mean(axis=1).sort_values(ascending=False).index
    pivot_m5 = pivot_m5.loc[row_order]

    fig, ax = plt.subplots(figsize=(10, max(5, len(pivot_m5) * 0.45)))
    sns.heatmap(
        pivot_m5,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        cbar_kws={"label": "mAP@5"},
        linewidths=0.5,
        ax=ax,
    )

    ax.set_title(
        f"{desc_name.upper()}: mAP@5 Heatmap across Bins and Similarity Measures",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Distance / Similarity Measure", fontweight="bold")
    ax.set_ylabel("Bin Allocation", fontweight="bold")

    out_path = out_dir / f"{desc_name}_measures_heatmap.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")



# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def load_all_results():
    """Load and merge all CSVs found in config_results/."""
    dfs = []
    for csv_file in CONFIG_DIR.glob("*.csv"):
        try:
            d = pd.read_csv(csv_file)
            dfs.append(d)
        except Exception as e:
            print(f"Warning: Could not read {csv_file.name}: {e}")
    if not dfs:
        raise FileNotFoundError(f"No experiment CSV files found in {CONFIG_DIR}")
    return pd.concat(dfs, ignore_index=True)


def main():
    setup_style()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_all_results()

    # 1. CIELab Visualizations
    if "lab" in df["descriptor"].values:
        print("Generating CIELab plots...")
        plot_distances_for_descriptor(df, PLOTS_DIR, desc_name="lab", target_bins="8-64-64")
        plot_top10_configurations(df, PLOTS_DIR, desc_name="lab")
        plot_measures_heatmap(df, PLOTS_DIR, desc_name="lab")

    # 2. HSV Visualizations
    if "hsv" in df["descriptor"].values:
        print("Generating HSV plots...")
        plot_distances_for_descriptor(df, PLOTS_DIR, desc_name="hsv", target_bins="16-16-8")
        plot_top10_configurations(df, PLOTS_DIR, desc_name="hsv")
        plot_measures_heatmap(df, PLOTS_DIR, desc_name="hsv")


if __name__ == "__main__":
    main()