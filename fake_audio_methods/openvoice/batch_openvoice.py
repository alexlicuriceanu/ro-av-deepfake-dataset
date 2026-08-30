import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Optional
import torch
from tqdm import tqdm
from huggingface_hub import snapshot_download
from openvoice import se_extractor
from openvoice.api import ToneColorConverter


def setup_openvoice(device: str = "cuda") -> ToneColorConverter:
    ckpt_dir = Path(
        snapshot_download(
            repo_id="myshell-ai/OpenVoiceV2",
            allow_patterns=["converter/*"],
        )
    )
    converter_dir = ckpt_dir / "converter"
    config_path = converter_dir / "config.json"
    ckpt_path = converter_dir / "checkpoint.pth"

    tone_color_converter = ToneColorConverter(str(config_path), device=device)
    tone_color_converter.load_ckpt(str(ckpt_path))
    return tone_color_converter


def safe_extract_se(
    audio_path: Path, converter: ToneColorConverter, target_dir: str
) -> Optional[torch.Tensor]:
    """
    Attempts to extract speaker embedding with VAD enabled.
    Falls back to vad=False if the audio segment is deemed too short by VAD.
    """
    try:
        se, _ = se_extractor.get_se(
            str(audio_path), converter, target_dir=target_dir, vad=True
        )
        if not isinstance(se, torch.Tensor):
            se = torch.tensor(se)
        return se
    except (AssertionError, Exception):
        try:
            se, _ = se_extractor.get_se(
                str(audio_path), converter, target_dir=target_dir, vad=False
            )
            if not isinstance(se, torch.Tensor):
                se = torch.tensor(se)
            return se
        except Exception:
            return None


def convert_voice(
    src_path: Path,
    tgt_paths: List[Path],
    out_path: Path,
    converter: ToneColorConverter,
    tau: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="openvoice_se_"))

    try:
        # 1. Extract source embedding with fallback
        src_se = safe_extract_se(src_path, converter, target_dir=str(temp_dir))
        if src_se is None:
            raise ValueError(f"Could not extract embedding for source audio: {src_path.name}")

        # 2. Extract valid target embeddings across donor clips
        tgt_ses = []
        for path in tgt_paths:
            se = safe_extract_se(path, converter, target_dir=str(temp_dir))
            if se is not None:
                tgt_ses.append(se)

        if not tgt_ses:
            raise ValueError(f"All target donor clips failed embedding extraction for {src_path.name}")

        if len(tgt_ses) == 1:
            target_se = tgt_ses[0]
        else:
            target_se = torch.mean(torch.stack(tgt_ses), dim=0)

        # 3. Tone Color Conversion
        converter.convert(
            audio_src_path=str(src_path),
            src_se=src_se,
            tgt_se=target_se,
            output_path=str(out_path),
            tau=tau,
            message=" ",
        )
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def save_manifest_safe(manifest: Dict, manifest_path: Path):
    temp_path = manifest_path.with_suffix(".json.tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    temp_path.replace(manifest_path)


def main():
    parser = argparse.ArgumentParser(description="Batch OpenVoice V2 Generation Pipeline")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to master_manifest.json")
    parser.add_argument("--out_dir", type=Path, required=True, help="Base directory for faked audio output")
    parser.add_argument("--tau", type=float, default=0.2, help="Sampling temperature for normalizing flow")
    args = parser.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    ov_tasks = {
        k: v for k, v in manifest.items()
        if v.get("audio_manipulation", {}).get("assigned_method") == "openvoice"
    }

    if not ov_tasks:
        print("[INFO] No OpenVoice tasks found in the manifest.")
        return

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Initializing model on {device}...")
    converter = setup_openvoice(device=device)

    print(f"[INFO] Starting batch processing for {len(ov_tasks)} OpenVoice clips...")

    skipped_count = 0
    pbar = tqdm(ov_tasks.items(), desc="Generating OpenVoice Fakes")

    for key, entry in pbar:
        manipulation_data = entry["audio_manipulation"]

        # Resume capability
        if "faked_audio_path" in manipulation_data:
            existing_out = Path(manipulation_data["faked_audio_path"])
            if existing_out.exists() and existing_out.stat().st_size > 0:
                continue

        src_path = Path(entry["audio_path"])
        tgt_paths = [Path(p) for p in manipulation_data["donor_audio_paths"]]
        video_id = entry["video_id"]
        out_path = args.out_dir / video_id / src_path.name

        pbar.set_postfix({"current": src_path.name, "skipped": skipped_count})

        try:
            convert_voice(
                src_path=src_path,
                tgt_paths=tgt_paths,
                out_path=out_path,
                converter=converter,
                tau=args.tau,
            )

            manifest[key]["audio_manipulation"]["faked_audio_path"] = str(out_path.resolve())
            save_manifest_safe(manifest, args.manifest)

        except Exception as e:
            skipped_count += 1
            tqdm.write(f"[WARN] Skipped {src_path.name}: {e}")
            continue

    print(f"[SUCCESS] Batch processing finished. Completed: {len(ov_tasks) - skipped_count}, Skipped: {skipped_count}")


if __name__ == "__main__":
    main()