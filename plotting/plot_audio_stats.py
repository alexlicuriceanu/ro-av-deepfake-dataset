import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_manifest_dataframe(manifest_path: Path) -> Tuple[pd.DataFrame, Dict]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    records = []
    for clip_id, entry in manifest.items():
        audio_meta = entry.get("audio_manipulation", {})
        records.append({
            "clip_id": clip_id,
            "video_id": entry["video_id"],
            "dialect": entry["dialect"],
            "environment": entry["environment"],
            "protocol_1_ood": entry["splits"]["protocol_1_ood"],
            "protocol_2_fewshot": entry["splits"]["protocol_2_fewshot"],
            "audio_method": audio_meta.get("assigned_method", "unassigned"),
            "num_donors": len(audio_meta.get("donor_clips", []))
        })

    df = pd.DataFrame(records)
    return df, manifest


def setup_plot_style():
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 0.8


def print_text_summary(df: pd.DataFrame):
    print("=" * 60)
    print("                DATASET SUMMARY METRICS")
    print("=" * 60)
    print(f"Total Clips:               {len(df)}")
    print(f"Total Unique Speakers:     {df['video_id'].nunique()}")
    print(f"Dialects Present:          {', '.join(sorted(df['dialect'].unique()))}")
    print(f"Environments:              {', '.join(sorted(df['environment'].unique()))}")
    print(f"Audio Faking Methods:      {', '.join(sorted(df['audio_method'].unique()))}")
    print("-" * 60)

    print("\n--- Audio Fake Method Breakdown per Dialect ---")
    method_by_dialect = pd.crosstab(df["dialect"], df["audio_method"], margins=True)
    print(method_by_dialect.to_string())

    print("\n--- Split Protocol 1 (Zero-Shot OOD) Breakdown ---")
    p1_counts = pd.crosstab(df["dialect"], df["protocol_1_ood"], margins=True)
    print(p1_counts.to_string())

    print("\n--- Split Protocol 2 (Few-Shot Adaptation) Breakdown ---")
    p2_counts = pd.crosstab(df["dialect"], df["protocol_2_fewshot"], margins=True)
    print(p2_counts.to_string())
    print("=" * 60)


def plot_methods_per_dialect(df: pd.DataFrame, output_dir: Path):
    plt.figure(figsize=(11, 6))
    ax = sns.countplot(
        data=df,
        x="dialect",
        hue="audio_method",
        palette="viridis",
        order=sorted(df["dialect"].unique())
    )
    plt.title("Audio Manipulation Methods per Regional Dialect", pad=15, fontweight="bold")
    plt.xlabel("Dialect", fontweight="bold")
    plt.ylabel("Number of Clips", fontweight="bold")
    plt.legend(title="Audio Method", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=20)

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{int(height)}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center", va="bottom",
                fontsize=8, rotation=0, xytext=(0, 2),
                textcoords="offset points"
            )

    plt.tight_layout()
    out_file = output_dir / "01_methods_per_dialect.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"[SAVED] {out_file}")


def plot_splits_distribution(df: pd.DataFrame, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # 1. Protocol 1 - Zero-Shot OOD
    p1_ct = pd.crosstab(df["dialect"], df["protocol_1_ood"])
    
    # Custom color mapping for clear semantic meaning
    p1_colors = {
        "train": "#4C72B0",
        "val": "#55A868",
        "test_id": "#C44E52",
        "test_ood_maramures": "#8172B3",
        "test_ood_rep_moldova": "#CCB974"
    }
    cols_p1 = [c for c in p1_colors.keys() if c in p1_ct.columns]
    p1_ct = p1_ct[cols_p1]

    p1_ct.plot(
        kind="bar",
        stacked=True,
        ax=axes[0],
        color=[p1_colors[c] for c in cols_p1],
        edgecolor="white",
        linewidth=0.8,
        width=0.65
    )
    axes[0].set_title("Protocol 1: Zero-Shot OOD Dialect Partitioning", fontweight="bold", pad=12)
    axes[0].set_xlabel("Dialect", fontweight="bold")
    axes[0].set_ylabel("Number of Clips", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].legend(title="Split", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # Add total count labels on top of bars
    for i, total in enumerate(p1_ct.sum(axis=1)):
        axes[0].text(i, total + 10, f"{int(total)}", ha="center", va="bottom", fontweight="bold", fontsize=9)

    # 2. Protocol 2 - Supervised Few-Shot
    p2_ct = pd.crosstab(df["dialect"], df["protocol_2_fewshot"])
    p2_colors = {
        "train": "#4C72B0",
        "val": "#55A868",
        "test_id": "#C44E52",
        "test_ood_maramures": "#8172B3",
        "test_ood_rep_moldova": "#CCB974"
    }
    cols_p2 = [c for c in p2_colors.keys() if c in p2_ct.columns]
    p2_ct = p2_ct[cols_p2]

    p2_ct.plot(
        kind="bar",
        stacked=True,
        ax=axes[1],
        color=[p2_colors[c] for c in cols_p2],
        edgecolor="white",
        linewidth=0.8,
        width=0.65
    )
    axes[1].set_title("Protocol 2: Supervised / Few-Shot Adaptation Splits", fontweight="bold", pad=12)
    axes[1].set_xlabel("Dialect", fontweight="bold")
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(title="Split", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    for i, total in enumerate(p2_ct.sum(axis=1)):
        axes[1].text(i, total + 10, f"{int(total)}", ha="center", va="bottom", fontweight="bold", fontsize=9)

    axes[0].set_ylim(0, max(p1_ct.sum(axis=1).max(), p2_ct.sum(axis=1).max()) * 1.12)

    plt.tight_layout()
    out_file = output_dir / "02_protocol_splits_distribution.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_file}")


def plot_environment_method_matrix(df: pd.DataFrame, output_dir: Path):
    plt.figure(figsize=(10, 6))
    ct = pd.crosstab(df["audio_method"], [df["dialect"], df["environment"]])
    
    # Flatten multi-index for clearer heatmap labeling
    ct.columns = [f"{col[0]}\n({col[1]})" for col in ct.columns]
    
    sns.heatmap(ct, annot=True, fmt="d", cmap="YlGnBu", cbar=True, linewidths=0.5)
    plt.title("Sample Allocation Matrix: Method vs. Dialect (Environment)", pad=15, fontweight="bold")
    plt.xlabel("Dialect and Environment Setting", fontweight="bold")
    plt.ylabel("Audio Synthesis Method", fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    
    plt.tight_layout()
    out_file = output_dir / "03_method_environment_matrix.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"[SAVED] {out_file}")


def plot_donor_source_dialect_transfer(manifest: Dict, output_dir: Path):
    """
    Analyzes and plots which dialects are donating timbre to which dialects.
    """
    clip_to_dialect = {entry["absolute_path"]: entry["dialect"] for entry in manifest.values()}

    transfer_records = []
    for entry in manifest.values():
        target_dialect = entry["dialect"]
        donors = entry.get("audio_manipulation", {}).get("donor_clips", [])
        for d_path in donors:
            donor_dialect = clip_to_dialect.get(d_path, "Unknown")
            transfer_records.append({
                "target_dialect": target_dialect,
                "donor_dialect": donor_dialect
            })

    if not transfer_records:
        return

    transfer_df = pd.DataFrame(transfer_records)
    matrix = pd.crosstab(transfer_df["target_dialect"], transfer_df["donor_dialect"])

    plt.figure(figsize=(9, 7))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=True, linewidths=0.5)
    plt.title("Voice Conversion Timbre Transfer Matrix\n(Rows: Target Speaker Dialect | Columns: Donor Voice Dialect)", pad=15, fontweight="bold")
    plt.xlabel("Donor Timbre Dialect", fontweight="bold")
    plt.ylabel("Target Speaker Dialect", fontweight="bold")
    
    plt.tight_layout()
    out_file = output_dir / "04_donor_dialect_transfer_matrix.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"[SAVED] {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Master Manifest Statistics and Analytics Plotter")
    parser.add_argument("--manifest", type=Path, default=Path("../master_manifest.json"), help="Path to master JSON manifest")
    parser.add_argument("--output_dir", type=Path, default=Path("../audio_plots"), help="Directory to save generated figures")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    df, raw_manifest = load_manifest_dataframe(args.manifest)

    # 1. Print formatted console stats
    print_text_summary(df)

    # 2. Render all plots
    print("\nGenerating publication-quality figures...")
    plot_methods_per_dialect(df, args.output_dir)
    plot_splits_distribution(df, args.output_dir)
    plot_environment_method_matrix(df, args.output_dir)
    plot_donor_source_dialect_transfer(raw_manifest, args.output_dir)

    print(f"\nAll plots saved successfully to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()