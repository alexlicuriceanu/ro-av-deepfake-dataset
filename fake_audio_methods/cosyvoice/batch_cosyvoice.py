import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict
import torch
import torchaudio
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
REPO_DIR = CURRENT_DIR / "CosyVoice"

if not REPO_DIR.exists():
    raise FileNotFoundError(f"Could not find CosyVoice repo at {REPO_DIR}")

sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "third_party" / "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import CosyVoice3


def save_manifest_safe(manifest: Dict, manifest_path: Path):
    """Atomic write to prevent manifest corruption on sudden exit."""
    temp_path = manifest_path.with_suffix(".json.tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    temp_path.replace(manifest_path)


def merge_and_save_donor_audio(tgt_paths: list[Path], output_path: Path):
    """Loads, resamples to 24k, concatenates donor clips, and writes to disk."""
    tensors = []
    target_sr = 24000
    for p in tgt_paths:
        wav, sr = torchaudio.load(str(p))
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        tensors.append(wav)

    combined = torch.cat(tensors, dim=-1)
    if combined.shape[-1] > target_sr * 25:
        combined = combined[:, :target_sr * 25]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), combined, target_sr)


def main():
    parser = argparse.ArgumentParser(description="Batch CosyVoice 3 0.5B Generation Pipeline")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to master_manifest.json")
    parser.add_argument("--model_dir", type=str, default="pretrained_models/Fun-CosyVoice3-0.5B")
    parser.add_argument("--out_dir", type=Path, required=True, help="Base directory for faked audio output")
    args = parser.parse_args()

    model_path = Path(args.model_dir)
    if not model_path.is_absolute() and not model_path.exists():
        model_path = REPO_DIR / args.model_dir

    print(f"[INFO] Initializing CosyVoice3 from {model_path}...")
    cosyvoice = CosyVoice3(str(model_path))

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Filter for CosyVoice tasks
    cosy_tasks = {
        k: v for k, v in manifest.items()
        if v.get("audio_manipulation", {}).get("assigned_method") == "cosyvoice"
    }

    if not cosy_tasks:
        print("[INFO] No CosyVoice tasks found in the manifest.")
        return

    print(f"[INFO] Starting batch processing for {len(cosy_tasks)} CosyVoice clips...")

    skipped_count = 0
    temp_donor_dir = Path(tempfile.mkdtemp(prefix="cosyvoice_donors_"))
    donor_cache: Dict[str, Path] = {}
    pbar = tqdm(cosy_tasks.items(), desc="Generating CosyVoice Fakes")

    try:
        for key, entry in pbar:
            manip_data = entry["audio_manipulation"]

            # Resume check
            if "faked_audio_path" in manip_data:
                existing_out = Path(manip_data["faked_audio_path"])
                if existing_out.exists() and existing_out.stat().st_size > 0:
                    continue

            source_path = Path(entry["audio_path"])
            tgt_paths = [Path(p) for p in manip_data["donor_audio_paths"]]
            donor_vid = manip_data.get("donor_video_id", "unknown_donor")
            out_path = args.out_dir / entry["video_id"] / source_path.name

            pbar.set_postfix({"current": source_path.name, "skipped": skipped_count})

            try:
                out_path.parent.mkdir(parents=True, exist_ok=True)

                # 1. Retrieve or generate cached donor file
                if donor_vid in donor_cache and donor_cache[donor_vid].exists():
                    cached_prompt_path = donor_cache[donor_vid]
                else:
                    cached_prompt_path = temp_donor_dir / f"{donor_vid}_prompt.wav"
                    merge_and_save_donor_audio(tgt_paths, cached_prompt_path)
                    donor_cache[donor_vid] = cached_prompt_path

                # 2. Run Voice Conversion
                outputs = []
                for output in cosyvoice.inference_vc(str(source_path), str(cached_prompt_path), stream=False):
                    outputs.append(output["tts_speech"])

                if not outputs:
                    raise RuntimeError("Inference returned no audio outputs.")

                final_audio = torch.cat(outputs, dim=-1)

                # 3. Save Output
                torchaudio.save(str(out_path), final_audio, cosyvoice.sample_rate)

                # 4. Update Manifest
                manifest[key]["audio_manipulation"]["faked_audio_path"] = str(out_path.resolve())
                save_manifest_safe(manifest, args.manifest)

            except Exception as e:
                skipped_count += 1
                tqdm.write(f"[WARN] Skipped {source_path.name} - Error: {str(e)}")
                continue

    finally:
        # Clean up temporary scratch folder
        shutil.rmtree(temp_donor_dir, ignore_errors=True)

    print(f"[SUCCESS] CosyVoice batch processing complete. Finished: {len(cosy_tasks) - skipped_count}, Skipped: {skipped_count}")


if __name__ == "__main__":
    main()