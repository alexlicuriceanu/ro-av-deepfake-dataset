import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


METHOD_MAP = {
    "real": "Real",
    "openvoice": "OpenVoice",
    "seed_vc": "Seed-VC",
    "cosyvoice": "CosyVoice",
}

DIALECT_MAP = {
    "muntenia": "Muntenia",
    "transilvania": "Transilvania",
    "moldova": "Moldova",
    "rep_moldova": "Rep. Moldova",
    "maramures": "Maramureș",
    "oltenia": "Oltenia",
}

ENV_MAP = {
    "itw": "In-the-Wild",
    "interview": "In-the-Wild",
    "studio": "Studio",
}

SPLIT_MAP = {
    "train": "Train",
    "val": "Validation",
    "test_id": "Test (In-Domain)",
    "test_ood_maramures": "Test OOD (Maramureș)",
    "test_ood_rep_moldova": "Test OOD (Rep. Moldova)",
}

METHOD_COLORS = {
    "Real": "#2D3748",
    "OpenVoice": "#2B6CB0",
    "Seed-VC": "#0D9488",
    "CosyVoice": "#D97706",
}

SPLIT_COLORS = {
    "Train": "#4A6572",
    "Validation": "#6096BA",
    "Test (In-Domain)": "#A94442",
    "Test OOD (Maramureș)": "#7840A8",
    "Test OOD (Rep. Moldova)": "#C08552",
}

ENV_COLORS = {
    "Studio": "#1D4ED8",
    "In-the-Wild": "#C2410C",
}


def normalize_dialect_str(val: str) -> str:
    val_lower = str(val).lower()
    for k, v in DIALECT_MAP.items():
        if k in val_lower:
            return v
    return str(val).capitalize()


def parse_metadata(video_id: str, entry: dict):
    raw_dialect = entry.get("dialect")
    if not raw_dialect:
        raw_dialect = normalize_dialect_str(video_id)
    else:
        raw_dialect = DIALECT_MAP.get(str(raw_dialect).lower(), str(raw_dialect))

    raw_env = entry.get("environment")
    if not raw_env:
        raw_env = "In-the-Wild" if ("itw" in video_id.lower() or "interview" in video_id.lower()) else "Studio"
    else:
        raw_env = ENV_MAP.get(str(raw_env).lower(), str(raw_env).capitalize())

    return raw_dialect, raw_env


def load_manifest_dataframe(manifest_path: Path) -> Tuple[pd.DataFrame, Dict]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    records = []
    for clip_id, entry in manifest.items():
        audio_meta = entry.get("audio_manipulation", {})
        raw_method = audio_meta.get("assigned_method", "unassigned")
        method = METHOD_MAP.get(raw_method.lower(), raw_method)
        dialect, env = parse_metadata(entry.get("video_id", clip_id), entry)

        splits_info = entry.get("splits", {})
        p1_split = splits_info.get("protocol_1_ood", "unknown")
        p2_split = splits_info.get("protocol_2_fewshot", "unknown")

        donors = audio_meta.get("donor_audio_paths") or audio_meta.get("donor_clips") or []

        records.append({
            "clip_id": clip_id,
            "video_id": entry.get("video_id", "unknown"),
            "dialect": dialect,
            "environment": env,
            "protocol_1_ood": SPLIT_MAP.get(str(p1_split).lower(), str(p1_split)),
            "protocol_2_fewshot": SPLIT_MAP.get(str(p2_split).lower(), str(p2_split)),
            "audio_method": method,
            "num_donors": len(donors)
        })

    return pd.DataFrame(records), manifest


def setup_plot_style():
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams.update({
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": "#CBD5E1",
        "axes.linewidth": 1.0,
        "grid.color": "#E2E8F0",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#CBD5E1",
    })


def plot_methods_per_dialect(df: pd.DataFrame, output_dir: Path):
    fig, ax = plt.subplots(figsize=(11, 7.2))
    
    preferred_order = ["OpenVoice", "Seed-VC", "CosyVoice"]
    methods = [m for m in preferred_order if m in df["audio_method"].unique()]
    dialect_order = sorted(df["dialect"].unique())

    sns.countplot(
        data=df,
        x="dialect",
        hue="audio_method",
        palette=METHOD_COLORS,
        order=dialect_order,
        hue_order=methods,
        ax=ax,
        edgecolor="white",
        linewidth=0.8,
        legend=False
    )
    
    fig.suptitle("Audio Manipulation Methods per Regional Dialect", fontsize=13, fontweight="bold", y=0.98)
    
    handles = [mpatches.Patch(color=METHOD_COLORS[m], label=m) for m in methods]
    fig.legend(
        handles=handles,
        labels=methods,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=len(methods),
        frameon=True,
        title="Synthesis Method",
        title_fontproperties={"weight": "bold", "size": 11},
        fontsize=11
    )

    ax.set_xlabel("Regional Dialect", fontweight="bold")
    ax.set_ylabel("Number of Clips", fontweight="bold")
    ax.tick_params(axis="x", rotation=20)

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{int(height)}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center", va="bottom",
                fontsize=9, rotation=0, xytext=(0, 3),
                textcoords="offset points"
            )

    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    plt.subplots_adjust(top=0.76, bottom=0.15)

    out_file = output_dir / "01_methods_per_dialect.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_file}")


def plot_splits_distribution(df: pd.DataFrame, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), sharey=True)

    fig.suptitle("Dataset Partitioning across Dialects", fontsize=13, fontweight="bold", y=0.98)

    preferred_split_order = ["Train", "Validation", "Test (In-Domain)", "Test OOD (Maramureș)", "Test OOD (Rep. Moldova)"]

    # Protocol 1
    p1_ct = pd.crosstab(df["dialect"], df["protocol_1_ood"])
    cols_p1 = [c for c in preferred_split_order if c in p1_ct.columns]
    p1_ct = p1_ct[cols_p1]

    p1_ct.plot(
        kind="bar",
        stacked=True,
        ax=axes[0],
        color=[SPLIT_COLORS.get(c, "#777777") for c in cols_p1],
        edgecolor="white",
        linewidth=0.8,
        width=0.65,
        legend=False
    )
    axes[0].set_title("Protocol 1: Zero-Shot OOD Partitioning", fontweight="bold", pad=12)
    axes[0].set_xlabel("Regional Dialect", fontweight="bold")
    axes[0].set_ylabel("Number of Clips", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=20)

    for i, total in enumerate(p1_ct.sum(axis=1)):
        axes[0].text(i, total + 10, f"{int(total)}", ha="center", va="bottom", fontsize=9)

    # Protocol 2
    p2_ct = pd.crosstab(df["dialect"], df["protocol_2_fewshot"])
    cols_p2 = [c for c in preferred_split_order if c in p2_ct.columns]
    p2_ct = p2_ct[cols_p2]

    p2_ct.plot(
        kind="bar",
        stacked=True,
        ax=axes[1],
        color=[SPLIT_COLORS.get(c, "#777777") for c in cols_p2],
        edgecolor="white",
        linewidth=0.8,
        width=0.65,
        legend=False
    )
    axes[1].set_title("Protocol 2: Supervised / Few-Shot Splits", fontweight="bold", pad=12)
    axes[1].set_xlabel("Regional Dialect", fontweight="bold")
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="x", rotation=20)

    for i, total in enumerate(p2_ct.sum(axis=1)):
        axes[1].text(i, total + 10, f"{int(total)}", ha="center", va="bottom", fontsize=9)

    axes[0].set_ylim(0, max(p1_ct.sum(axis=1).max(), p2_ct.sum(axis=1).max()) * 1.15)

    all_present_splits = list(dict.fromkeys(cols_p1 + cols_p2))
    split_handles = [mpatches.Patch(color=SPLIT_COLORS.get(c, "#777777"), label=c) for c in all_present_splits]

    fig.legend(
        handles=split_handles,
        labels=all_present_splits,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=len(all_present_splits),
        frameon=True,
        fontsize=11,
        title="Split Partition",
        title_fontproperties={"weight": "bold", "size": 11}
    )

    plt.subplots_adjust(top=0.74, bottom=0.15, wspace=0.18)

    out_file = output_dir / "02_protocol_splits_distribution.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_file}")


def plot_environment_method_matrix(df: pd.DataFrame, output_dir: Path):
    plt.figure(figsize=(11, 6))
    ct = pd.crosstab(df["audio_method"], [df["dialect"], df["environment"]])
    ct.columns = [f"{col[0]}\n({col[1]})" for col in ct.columns]
    
    # Custom academic blue-to-teal sequential colormap
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", cbar=True, linewidths=0.8, linecolor="white")
    plt.title("Sample Allocation: Method vs. Dialect (Environment)", pad=15, fontweight="bold")
    plt.xlabel("Dialect & Environment Setting", fontweight="bold")
    plt.ylabel("Synthesis Method", fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    
    plt.tight_layout()
    out_file = output_dir / "03_method_environment_matrix.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_file}")


def plot_donor_source_dialect_transfer(manifest: Dict, output_dir: Path):
    transfer_records = []

    for key, entry in manifest.items():
        src_dialect = DIALECT_MAP.get(str(entry.get("dialect", "")).lower(), normalize_dialect_str(entry.get("video_id", key)))
        manip = entry.get("audio_manipulation", {})

        donor_d = manip.get("donor_dialect")
        if donor_d:
            transfer_records.append({
                "target_dialect": src_dialect,
                "donor_dialect": DIALECT_MAP.get(str(donor_d).lower(), normalize_dialect_str(donor_d))
            })
            continue

        donor_vid = manip.get("donor_video_id")
        if donor_vid:
            transfer_records.append({
                "target_dialect": src_dialect,
                "donor_dialect": normalize_dialect_str(donor_vid)
            })
            continue

        donors = manip.get("donor_audio_paths") or manip.get("donor_clips") or []
        for d_path in donors:
            transfer_records.append({
                "target_dialect": src_dialect,
                "donor_dialect": normalize_dialect_str(Path(d_path).parent.name)
            })

    if not transfer_records:
        return

    transfer_df = pd.DataFrame(transfer_records)
    matrix = pd.crosstab(transfer_df["target_dialect"], transfer_df["donor_dialect"])

    plt.figure(figsize=(9, 7))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=True, linewidths=0.8, linecolor="white")
    plt.title("Voice Conversion Timbre Transfer Matrix\n(Rows: Target Dialect | Columns: Donor Timbre Dialect)", pad=15, fontweight="bold")
    plt.xlabel("Donor Timbre Dialect", fontweight="bold")
    plt.ylabel("Target Speaker Dialect", fontweight="bold")
    plt.xticks(rotation=25)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    out_file = output_dir / "04_donor_dialect_transfer_matrix.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
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

    plot_methods_per_dialect(df, args.output_dir)
    plot_splits_distribution(df, args.output_dir)
    plot_environment_method_matrix(df, args.output_dir)
    plot_donor_source_dialect_transfer(raw_manifest, args.output_dir)

    print(f"[SUCCESS] Harmonized plots generated at: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()