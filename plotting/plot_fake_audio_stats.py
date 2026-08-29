import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
import librosa
import matplotlib
matplotlib.use('Agg')  # Headless mode for cluster/SSH environments
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

warnings.simplefilter('ignore')

# ==========================================
# NORMALIZATION MAPS & REFINED PALETTE
# ==========================================
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
}

ENV_MAP = {
    "itw": "In-the-Wild",
    "interview": "In-the-Wild",
    "studio": "Studio",
}

# Refined academic palette with softer Slate for Real (no harsh darks)
METHOD_COLORS = {
    "Real": "#2D3748",
    "OpenVoice": "#2B6CB0",
    "Seed-VC": "#0D9488",
    "CosyVoice": "#D97706",
}


# Standardized styling parameters for all boxplots
BOXPLOT_KWARGS = {
    "linewidth": 0.8,
    "fliersize": 2.5,
    "flierprops": {
        "marker": "o",
        "markersize": 2.5,
        "markerfacecolor": "#94A3B8",
        "markeredgecolor": "#64748B",
        "markeredgewidth": 0.4,
        "alpha": 0.6
    },
    "boxprops": {"edgecolor": "#334155", "linewidth": 0.8, "alpha": 0.9},
    "whiskerprops": {"color": "#475569", "linewidth": 0.8},
    "capprops": {"color": "#475569", "linewidth": 0.8},
    "medianprops": {"color": "#0F172A", "linewidth": 1.1},
}


def setup_plot_style():
    """Configures clean typography, subtle grids, and minimal spines."""
    sns.set_theme(style="whitegrid", font_scale=1.0)
    plt.rcParams.update({
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "text.color": "#000000",
        "axes.labelcolor": "#000000",
        "axes.edgecolor": "#CBD5E1",
        "axes.linewidth": 0.7,
        "grid.color": "#F1F5F9",
        "grid.linestyle": "-",
        "grid.alpha": 0.9,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#E2E8F0",
    })


def parse_metadata(video_id: str, entry: dict):
    """Extracts standardized dialect and acoustic environment."""
    raw_dialect = entry.get("dialect")
    if not raw_dialect:
        vid_lower = video_id.lower()
        if "rep_moldova" in vid_lower:
            raw_dialect = "rep_moldova"
        elif "moldova" in vid_lower:
            raw_dialect = "moldova"
        elif "maramures" in vid_lower:
            raw_dialect = "maramures"
        elif "transilvania" in vid_lower or "ardelean" in vid_lower:
            raw_dialect = "transilvania"
        elif "banat" in vid_lower:
            raw_dialect = "banat"
        elif "oltenia" in vid_lower:
            raw_dialect = "oltenia"
        elif "muntenia" in vid_lower:
            raw_dialect = "muntenia"
        elif "dobrogea" in vid_lower:
            raw_dialect = "dobrogea"
        else:
            raw_dialect = "Other"

    raw_env = entry.get("environment")
    if not raw_env:
        raw_env = "itw" if ("itw" in video_id.lower() or "interview" in video_id.lower()) else "studio"

    dialect = DIALECT_MAP.get(str(raw_dialect).lower(), str(raw_dialect))
    env = ENV_MAP.get(str(raw_env).lower(), str(raw_env).capitalize())

    return dialect, env


def annotate_bars(ax: plt.Axes, is_float: bool = False, unit: str = ""):
    """Helper to cleanly annotate values above bars with regular font weight."""
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            val_str = f"{height:.2f}{unit}" if is_float else f"{int(height)}{unit}"
            ax.annotate(
                val_str,
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center", va="bottom",
                fontsize=8.5, fontweight="normal",
                color="#334155",
                xytext=(0, 2), textcoords="offset points"
            )


# ==========================================
# SPEAKER EMBEDDING EXTRACTOR (ECAPA / WavLM)
# ==========================================
class SpeakerVerificationModel:
    """Extracts speaker embeddings using SpeechBrain ECAPA-TDNN or fallback WavLM."""
    def __init__(self, device: torch.device):
        self.device = device
        self.model_type = None
        
        try:
            from speechbrain.inference.speaker import SpeakerRecognition
            self.model = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="./checkpoints/spkrec_ecapa",
                run_opts={"device": str(device)}
            )
            self.model_type = "speechbrain"
        except Exception:
            try:
                from transformers import AutoFeatureExtractor, AutoModel
                self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
                self.model = AutoModel.from_pretrained("microsoft/wavlm-base-plus-sv").to(device).eval()
                self.model_type = "wavlm"
            except Exception as e:
                print(f"[WARN] Could not load dedicated speaker model ({e}). Falling back to MFCC timbre vectors.")
                self.model_type = "mfcc"

    @torch.no_grad()
    def extract_embedding(self, wav_tensor: torch.Tensor, sr: int = 16000) -> np.ndarray:
        if self.model_type == "speechbrain":
            emb = self.model.encode_batch(wav_tensor.to(self.device))
            return emb.squeeze().cpu().numpy()
        elif self.model_type == "wavlm":
            inputs = self.feature_extractor(wav_tensor.squeeze().numpy(), sampling_rate=sr, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1)
            return emb.squeeze().cpu().numpy()
        else:
            mfcc = librosa.feature.mfcc(y=wav_tensor.squeeze().numpy(), sr=sr, n_mfcc=40)
            return np.mean(mfcc, axis=1)


def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    dot = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))


def compute_acoustic_features(wav: np.ndarray, sr: int) -> Dict[str, float]:
    """Computes RMS Energy, Dynamic Range, Spectral Centroid, and Spectral Roll-off."""
    if len(wav) == 0:
        return {}
    
    rms = float(np.sqrt(np.mean(wav**2)))
    rms_db = float(20 * np.log10(rms + 1e-9))
    peak = float(np.max(np.abs(wav)))
    crest_factor_db = float(20 * np.log10((peak + 1e-9) / (rms + 1e-9)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=wav, sr=sr)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=wav, sr=sr, roll_percent=0.85)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=wav)))

    return {
        "rms_db": rms_db,
        "crest_factor_db": crest_factor_db,
        "spectral_centroid": centroid,
        "spectral_rolloff": rolloff,
        "zero_crossing_rate": zcr
    }


def load_and_resample(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, torch.Tensor]:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0).numpy(), wav


def main():
    parser = argparse.ArgumentParser(description="Merged Dataset EDA & Acoustic Audio Statistics Analyzer")
    
    # Execution Flags
    parser.add_argument("--run-eda", action="store_true", help="Run the dataset exploratory data analysis (EDA) plots.")
    parser.add_argument("--run-audio-analysis", action="store_true", help="Run the audio quality metrics and speaker similarity analysis.")
    
    # Input/Output Configs
    parser.add_argument("--manifest", type=Path, required=True, help="Path to master_manifest.json")
    parser.add_argument("--out_dir", type=Path, default=Path("./analysis_outputs"), help="Directory to save figures and CSVs")
    
    # Custom File Names
    parser.add_argument("--eda-plot-name", type=str, default="audio_dataset_statistics.png", help="Filename for dataset EDA plot.")
    parser.add_argument("--audio-plot-name", type=str, default="acoustic_quality_and_speaker_similarity.png", help="Filename for audio quality & similarity plot.")
    parser.add_argument("--metrics-csv-name", type=str, default="acoustic_and_similarity_metrics.csv", help="Filename for raw acoustic metrics CSV.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional sample limit for quick test runs on audio analysis")
    
    args = parser.parse_args()

    if not args.run_eda and not args.run_audio_analysis:
        print("[INFO] No analysis flags provided. Please specify at least one: --run-eda or --run-audio-analysis")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # ==========================================
    # 1. RUN DATASET EDA PIPELINE
    # ==========================================
    if args.run_eda:
        print("\n[INFO] Starting Dataset EDA Analysis...")
        records_eda = []
        missing_files_eda = 0

        for key, entry in tqdm(manifest.items(), desc="Analyzing Manifest for EDA"):
            manip = entry.get("audio_manipulation", {})
            raw_method = manip.get("assigned_method", "unassigned")
            faked_path_str = manip.get("faked_audio_path")

            if not faked_path_str:
                continue

            faked_path = Path(faked_path_str)
            if not faked_path.exists() or faked_path.stat().st_size == 0:
                missing_files_eda += 1
                continue

            video_id = entry.get("video_id", faked_path.parent.name)
            dialect, env = parse_metadata(video_id, entry)
            method = METHOD_MAP.get(raw_method.lower(), raw_method)

            try:
                info = sf.info(str(faked_path))
                duration = info.duration
                sr = info.samplerate
                channels = info.channels
            except Exception:
                missing_files_eda += 1
                continue

            records_eda.append({
                "key": key,
                "method": method,
                "dialect": dialect,
                "environment": env,
                "duration": duration,
                "sample_rate": sr,
                "channels": channels,
                "video_id": video_id
            })

        if records_eda:
            df_eda = pd.DataFrame(records_eda)
            method_order = [m for m in ["OpenVoice", "Seed-VC", "CosyVoice"] if m in df_eda["method"].unique()]
            if not method_order:
                method_order = df_eda["method"].value_counts().index.tolist()

            dialect_order = sorted(df_eda["dialect"].unique())

            # Terminal Printout for EDA
            print("\n" + "=" * 65)
            print("                    DATASET AUDIT SUMMARY                    ")
            print("=" * 65)
            print(f"Total Processed Fakes  : {len(df_eda):,}")
            print(f"Total Audio Duration   : {df_eda['duration'].sum() / 3600:.2f} hours ({df_eda['duration'].sum() / 60:.1f} mins)")
            print(f"Mean Clip Duration     : {df_eda['duration'].mean():.2f}s (± {df_eda['duration'].std():.2f}s)")
            print(f"Missing / Broken Clips : {missing_files_eda}")
            print("=" * 65 + "\n")

            fig, axes = plt.subplots(2, 3, figsize=(22, 13))

            # Main Figure Title
            fig.suptitle("Dataset Distribution & Dialect Representation", fontsize=14, fontweight="bold", y=0.985)

            # Subplot 1: Total Clips per Method
            ax1 = axes[0, 0]
            sns.countplot(
                data=df_eda,
                x="method",
                ax=ax1,
                palette=METHOD_COLORS,
                order=method_order,
                edgecolor="#CBD5E1",
                linewidth=0.7
            )
            ax1.set_title("Total Clips per Method", pad=12, fontsize=12)
            ax1.set_xlabel("Synthesis Method")
            ax1.set_ylabel("Clip Count")
            annotate_bars(ax1)
            ax1.set_ylim(0, ax1.get_ylim()[1] * 1.12)

            # Subplot 2: Dialect Representation by Method
            ax2 = axes[0, 1]
            sns.countplot(
                data=df_eda,
                x="dialect",
                hue="method",
                ax=ax2,
                palette=METHOD_COLORS,
                order=dialect_order,
                hue_order=method_order,
                edgecolor="#CBD5E1",
                linewidth=0.7,
                legend=False
            )
            ax2.set_title("Dialect Representation by Method", pad=12, fontsize=12)
            ax2.set_xlabel("Regional Dialect")
            ax2.set_ylabel("Clip Count")
            ax2.tick_params(axis='x', rotation=20)
            annotate_bars(ax2)
            ax2.set_ylim(0, ax2.get_ylim()[1] * 1.15)

            # Subplot 3: Total Hours per Dialect
            ax3 = axes[0, 2]
            hours_df = df_eda.groupby(["dialect", "method"])["duration"].sum().reset_index()
            hours_df["hours"] = hours_df["duration"] / 3600
            sns.barplot(
                data=hours_df,
                x="dialect",
                y="hours",
                hue="method",
                ax=ax3,
                palette=METHOD_COLORS,
                order=dialect_order,
                hue_order=method_order,
                edgecolor="#CBD5E1",
                linewidth=0.7,
                legend=False
            )
            ax3.set_title("Total Duration (Hours) by Dialect", pad=12, fontsize=12)
            ax3.set_xlabel("Regional Dialect")
            ax3.set_ylabel("Total Duration (Hours)")
            ax3.tick_params(axis='x', rotation=20)
            annotate_bars(ax3, is_float=True, unit="h")
            ax3.set_ylim(0, ax3.get_ylim()[1] * 1.15)

            # Subplot 4: Clip Duration Distribution
            ax4 = axes[1, 0]
            sns.boxplot(
                data=df_eda,
                x="method",
                y="duration",
                ax=ax4,
                palette=METHOD_COLORS,
                order=method_order,
                **BOXPLOT_KWARGS
            )
            ax4.set_title("Clip Duration Distribution by Method", pad=12, fontsize=12)
            ax4.set_xlabel("Synthesis Method")
            ax4.set_ylabel("Duration (Seconds)")

            # Subplot 5: Duration Density Curves
            ax5 = axes[1, 1]
            sns.kdeplot(
                data=df_eda,
                x="duration",
                hue="method",
                ax=ax5,
                fill=True,
                common_norm=False,
                palette=METHOD_COLORS,
                hue_order=method_order,
                alpha=0.25,
                linewidth=1.2,
                legend=False
            )
            ax5.set_title("Duration Density Curves", pad=12, fontsize=12)
            ax5.set_xlabel("Duration (Seconds)")
            ax5.set_ylabel("Density")

            # Subplot 6: Acoustic Environment Breakdown
            ax6 = axes[1, 2]
            env_df = df_eda.groupby(["environment", "method"]).size().reset_index(name="count")
            env_order = [e for e in ["Studio", "In-the-Wild"] if e in env_df["environment"].unique()]
            sns.barplot(
                data=env_df,
                x="environment",
                y="count",
                hue="method",
                ax=ax6,
                palette=METHOD_COLORS,
                order=env_order,
                hue_order=method_order,
                edgecolor="#CBD5E1",
                linewidth=0.7,
                legend=False
            )
            ax6.set_title("Acoustic Environment Distribution", pad=12, fontsize=12)
            ax6.set_xlabel("Acoustic Domain")
            ax6.set_ylabel("Clip Count")
            annotate_bars(ax6)
            ax6.set_ylim(0, ax6.get_ylim()[1] * 1.12)

            # Clean top legend
            eda_handles = [mpatches.Patch(color=METHOD_COLORS[m], label=m) for m in method_order]
            fig.legend(
                handles=eda_handles,
                labels=method_order,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.935),
                ncol=len(method_order),
                frameon=True,
                fontsize=11,
                title="Synthesis Method",
                title_fontproperties={"weight": "bold", "size": 11}
            )

            plt.subplots_adjust(top=0.84, bottom=0.08, hspace=0.38, wspace=0.26)

            out_img_eda = args.out_dir / args.eda_plot_name
            plt.savefig(out_img_eda, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"[SUCCESS] Dataset EDA plot saved to: {out_img_eda.resolve()}")

    # ==========================================
    # 2. RUN AUDIO QUALITY & SIMILARITY ANALYSIS
    # ==========================================
    if args.run_audio_analysis:
        print("\n[INFO] Starting Audio Quality & Speaker Similarity Analysis...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Initializing embedding model on {device}...")
        spk_verifier = SpeakerVerificationModel(device)

        records_audio = []
        donor_emb_cache: Dict[str, np.ndarray] = {}

        keys = list(manifest.keys())
        if args.max_samples:
            keys = keys[:args.max_samples]

        for k in tqdm(keys, desc="Evaluating Audio Metrics"):
            entry = manifest[k]
            manip = entry.get("audio_manipulation", {})
            raw_method = manip.get("assigned_method", "unassigned")
            faked_path_str = manip.get("faked_audio_path")
            real_path_str = entry.get("audio_path")
            donor_paths = manip.get("donor_audio_paths", [])
            donor_vid = manip.get("donor_video_id", "unknown_donor")

            # 1. Real Audio Metrics
            if real_path_str and Path(real_path_str).exists():
                try:
                    real_np, _ = load_and_resample(real_path_str, 16000)
                    real_feats = compute_acoustic_features(real_np, 16000)
                    records_audio.append({
                        "clip_id": k,
                        "type": "Real",
                        "method": "Real",
                        "speaker_similarity": 1.0,
                        **real_feats
                    })
                except Exception:
                    pass

            # 2. Faked Audio Metrics & Similarity against Donor
            if faked_path_str and Path(faked_path_str).exists() and Path(faked_path_str).stat().st_size > 0:
                try:
                    fake_np, fake_tensor = load_and_resample(faked_path_str, 16000)
                    fake_feats = compute_acoustic_features(fake_np, 16000)
                    fake_emb = spk_verifier.extract_embedding(fake_tensor, 16000)

                    if donor_vid in donor_emb_cache:
                        donor_emb = donor_emb_cache[donor_vid]
                    else:
                        valid_donors = [p for p in donor_paths if Path(p).exists()]
                        if valid_donors:
                            d_tensors = [load_and_resample(p, 16000)[1] for p in valid_donors]
                            combined_donor = torch.cat(d_tensors, dim=-1)
                            donor_emb = spk_verifier.extract_embedding(combined_donor, 16000)
                            donor_emb_cache[donor_vid] = donor_emb
                        else:
                            donor_emb = None

                    sim = compute_cosine_similarity(fake_emb, donor_emb) if donor_emb is not None else np.nan
                    method_name = METHOD_MAP.get(raw_method.lower(), raw_method)

                    records_audio.append({
                        "clip_id": k,
                        "type": "Fake",
                        "method": method_name,
                        "speaker_similarity": sim,
                        **fake_feats
                    })
                except Exception:
                    continue

        if records_audio:
            df_audio = pd.DataFrame(records_audio)
            csv_out = args.out_dir / args.metrics_csv_name
            df_audio.to_csv(csv_out, index=False)
            print(f"[INFO] Raw audio metrics CSV saved to: {csv_out.resolve()}")

            method_order_audio = ["Real", "OpenVoice", "Seed-VC", "CosyVoice"]
            method_order_audio = [m for m in method_order_audio if m in df_audio["method"].unique()]

            fig, axes = plt.subplots(2, 3, figsize=(22, 13))

            # Main Figure Title
            fig.suptitle("Acoustic Quality Metrics & Speaker Timbre Transfer", fontsize=14, fontweight="bold", y=0.985)

            # Subplot 1: Speaker Similarity Boxplot
            ax1 = axes[0, 0]
            fake_df = df_audio[df_audio["type"] == "Fake"].dropna(subset=["speaker_similarity"])
            fake_methods = [m for m in method_order_audio if m != "Real"]
            sns.boxplot(
                data=fake_df,
                x="method",
                y="speaker_similarity",
                ax=ax1,
                palette=METHOD_COLORS,
                order=fake_methods,
                **BOXPLOT_KWARGS
            )
            ax1.set_title("Speaker Similarity to Donor Prompt", pad=12, fontsize=12)
            ax1.set_xlabel("Synthesis Method")
            ax1.set_ylabel("Cosine Similarity")

            # Subplot 2: Speaker Similarity KDE
            ax2 = axes[0, 1]
            sns.kdeplot(
                data=fake_df,
                x="speaker_similarity",
                hue="method",
                ax=ax2,
                fill=True,
                common_norm=False,
                palette=METHOD_COLORS,
                hue_order=fake_methods,
                alpha=0.25,
                linewidth=1.2,
                legend=False
            )
            ax2.set_title("Speaker Similarity Distribution (KDE)", pad=12, fontsize=12)
            ax2.set_xlabel("Cosine Similarity")
            ax2.set_ylabel("Density")

            # Subplot 3: Spectral Roll-off Frequency
            ax3 = axes[0, 2]
            sns.boxplot(
                data=df_audio,
                x="method",
                y="spectral_rolloff",
                ax=ax3,
                palette=METHOD_COLORS,
                order=method_order_audio,
                **BOXPLOT_KWARGS
            )
            ax3.set_title("Spectral Roll-off (85% Energy Cutoff)", pad=12, fontsize=12)
            ax3.set_xlabel("Synthesis Method")
            ax3.set_ylabel("Frequency (Hz)")

            # Subplot 4: Spectral Centroid
            ax4 = axes[1, 0]
            sns.boxplot(
                data=df_audio,
                x="method",
                y="spectral_centroid",
                ax=ax4,
                palette=METHOD_COLORS,
                order=method_order_audio,
                **BOXPLOT_KWARGS
            )
            ax4.set_title("Spectral Centroid Distribution", pad=12, fontsize=12)
            ax4.set_xlabel("Synthesis Method")
            ax4.set_ylabel("Centroid Frequency (Hz)")

            # Subplot 5: RMS Energy Level
            ax5 = axes[1, 1]
            sns.kdeplot(
                data=df_audio,
                x="rms_db",
                hue="method",
                ax=ax5,
                fill=True,
                common_norm=False,
                palette=METHOD_COLORS,
                hue_order=method_order_audio,
                alpha=0.25,
                linewidth=1.2,
                legend=False
            )
            ax5.set_title("RMS Energy Level Distribution", pad=12, fontsize=12)
            ax5.set_xlabel("RMS Energy (dBFS)")
            ax5.set_ylabel("Density")

            # Subplot 6: Dynamic Range / Crest Factor
            ax6 = axes[1, 2]
            sns.boxplot(
                data=df_audio,
                x="method",
                y="crest_factor_db",
                ax=ax6,
                palette=METHOD_COLORS,
                order=method_order_audio,
                **BOXPLOT_KWARGS
            )
            ax6.set_title("Dynamic Range (Crest Factor)", pad=12, fontsize=12)
            ax6.set_xlabel("Synthesis Method")
            ax6.set_ylabel("Peak-to-RMS Ratio (dB)")

            # Clean top legend
            audio_handles = [mpatches.Patch(color=METHOD_COLORS[m], label=m) for m in method_order_audio]
            fig.legend(
                handles=audio_handles,
                labels=method_order_audio,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.935),
                ncol=len(method_order_audio),
                frameon=True,
                fontsize=11,
                title="Class / Method",
                title_fontproperties={"weight": "bold", "size": 11}
            )

            plt.subplots_adjust(top=0.84, bottom=0.08, hspace=0.38, wspace=0.26)

            img_out_audio = args.out_dir / args.audio_plot_name
            plt.savefig(img_out_audio, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"[SUCCESS] Audio quality & similarity plot saved to: {img_out_audio.resolve()}")


if __name__ == "__main__":
    main()