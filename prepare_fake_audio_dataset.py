import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict


def resolve_relative_path(target_path: str | Path, base_root: Path) -> str:
    """Converts an absolute path to a relative path against base_root if possible."""
    try:
        return str(Path(target_path).resolve().relative_to(base_root.resolve()))
    except ValueError:
        # Fallback to os.path.relpath if on different mount/drive or not direct child
        return os.path.relpath(str(target_path), str(base_root))


def main():
    parser = argparse.ArgumentParser(description="Prepare Audio Dataset Metadata Manifest")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to master_manifest.json"
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=None,
        help="Base root directory to compute relative paths from (defaults to manifest parent directory)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./fake_audio_metadata.json"),
        help="Output path for the generated audio metadata JSON"
    )
    parser.add_argument(
        "--check_files",
        action="store_true",
        help="Verify file existence on disk before including in metadata"
    )
    args = parser.parse_args()

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    dataset_root = args.dataset_root.resolve() if args.dataset_root else args.manifest.resolve().parent

    with open(args.manifest, "r", encoding="utf-8") as f:
        master_manifest: Dict[str, Any] = json.load(f)

    audio_metadata: Dict[str, Any] = {}
    real_count = 0
    fake_count = 0
    missing_real = 0
    missing_fake = 0

    print(f"[INFO] Building audio metadata relative to base root: {dataset_root}")

    for master_key, entry in master_manifest.items():
        clip_id = entry.get("clip_id", "")
        video_id = entry.get("video_id", "")
        dialect = entry.get("dialect", "unknown")
        environment = entry.get("environment", "unknown")
        splits = entry.get("splits", {})

        # ==========================================
        # 1. REAL AUDIO ENTRY
        # ==========================================
        real_audio_path_str = entry.get("audio_path")
        if real_audio_path_str:
            real_audio_path = Path(real_audio_path_str)
            if args.check_files and (not real_audio_path.exists() or real_audio_path.stat().st_size == 0):
                missing_real += 1
            else:
                real_entry_name = f"{master_key}__real"
                audio_metadata[real_entry_name] = {
                    "key": master_key,
                    "clip_id": clip_id,
                    "video_id": video_id,
                    "path": resolve_relative_path(real_audio_path, dataset_root),
                    "dialect": dialect,
                    "environment": environment,
                    "splits": splits,
                    "type": "real",
                    "faking_method": None
                }
                real_count += 1

        # ==========================================
        # 2. FAKE AUDIO ENTRY
        # ==========================================
        audio_manip = entry.get("audio_manipulation", {})
        fake_audio_path_str = audio_manip.get("faked_audio_path")
        assigned_method = audio_manip.get("assigned_method")

        if fake_audio_path_str and assigned_method:
            fake_audio_path = Path(fake_audio_path_str)
            if args.check_files and (not fake_audio_path.exists() or fake_audio_path.stat().st_size == 0):
                missing_fake += 1
            else:
                fake_entry_name = f"{master_key}__fake_{assigned_method}"
                audio_metadata[fake_entry_name] = {
                    "key": master_key,
                    "clip_id": clip_id,
                    "video_id": video_id,
                    "path": resolve_relative_path(fake_audio_path, dataset_root),
                    "dialect": dialect,
                    "environment": environment,
                    "splits": splits,
                    "type": "fake",
                    "faking_method": assigned_method
                }
                fake_count += 1

    # Save output JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_out = args.output.with_suffix(".json.tmp")
    with open(temp_out, "w", encoding="utf-8") as f:
        json.dump(audio_metadata, f, indent=2, ensure_ascii=False)
    temp_out.replace(args.output)

    # Print summary
    print("\n" + "=" * 60)
    print("             AUDIO DATASET METADATA SUMMARY             ")
    print("=" * 60)
    print(f"Total Master Manifest Keys : {len(master_manifest):,}")
    print(f"Total Real Audio Entries   : {real_count:,}")
    print(f"Total Fake Audio Entries   : {fake_count:,}")
    print(f"Total Audio Samples in Set : {len(audio_metadata):,}")
    if args.check_files:
        print(f"Missing Real Audio Files   : {missing_real}")
        print(f"Missing Fake Audio Files   : {missing_fake}")
    print(f"Metadata Saved To          : {args.output.resolve()}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()