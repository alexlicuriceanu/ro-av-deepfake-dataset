import argparse
import shutil
from pathlib import Path
from typing import List, Optional
from huggingface_hub import HfApi, create_repo


def upload_dataset(
    repo_id: str,
    dataset_root: Path,
    folders: List[str],
    metadata_files: Optional[List[str]] = None,
    primary_metadata_csv: Optional[str] = None,
    private: bool = False,
):
    """
    Modular uploader for audio, video, or multimodal datasets to Hugging Face.
    Handles temporary metadata.csv generation for Hub auto-discovery and cleans up on exit.
    """
    api = HfApi()
    root = dataset_root.resolve()

    if not root.exists():
        raise FileNotFoundError(f"Dataset root directory not found: {root}")

    print(f"[INFO] Initializing repository: {repo_id} (Private={private})...")
    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True
    )

    created_temp_metadata_csv = False
    temp_metadata_target = root / "metadata.csv"

    try:
        # 1. Handle metadata.csv generation for HF web auto-discovery
        if primary_metadata_csv:
            source_meta = root / primary_metadata_csv
            if not source_meta.exists():
                raise FileNotFoundError(f"Specified primary metadata file not found: {source_meta}")

            if not temp_metadata_target.exists():
                print(f"[INFO] Creating temporary 'metadata.csv' from {source_meta.name} for Hub discovery...")
                shutil.copyfile(source_meta, temp_metadata_target)
                created_temp_metadata_csv = True

        # 2. Upload Metadata Files to the root of the repo
        all_metadata_to_upload = list(metadata_files) if metadata_files else []
        if (created_temp_metadata_csv or temp_metadata_target.exists()) and "metadata.csv" not in all_metadata_to_upload:
            all_metadata_to_upload.append("metadata.csv")

        for meta_name in all_metadata_to_upload:
            meta_path = root / meta_name
            if meta_path.exists():
                print(f"[INFO] Uploading metadata file: {meta_name}...")
                api.upload_file(
                    path_or_fileobj=str(meta_path),
                    path_in_repo=meta_name,
                    repo_id=repo_id,
                    repo_type="dataset"
                )
            else:
                print(f"[WARN] Metadata file '{meta_name}' not found in {root}. Skipping.")

        # 3. Upload Data Folders
        for rel_folder in folders:
            folder_path = root / rel_folder
            if not folder_path.exists():
                print(f"[WARN] Directory '{folder_path}' does not exist. Skipping.")
                continue

            print(f"[INFO] Uploading folder '{rel_folder}' ({folder_path})...")
            api.upload_folder(
                folder_path=str(folder_path),
                path_in_repo=rel_folder,
                repo_id=repo_id,
                repo_type="dataset",
                ignore_patterns=[
                    "**/.cache/**",
                    "**/*.tmp",
                    "**/*.pyc",
                    "**/__pycache__/**",
                    "**/.git/**"
                ]
            )

        print(f"\n[SUCCESS] Dataset successfully published at: https://huggingface.co/datasets/{repo_id}")

    finally:
        # 4. Clean up temporary metadata.csv
        if created_temp_metadata_csv and temp_metadata_target.exists():
            print("[INFO] Cleaning up local temporary 'metadata.csv'...")
            temp_metadata_target.unlink()


def main():
    parser = argparse.ArgumentParser(description="Modular Hugging Face Dataset Uploader (Audio / Video / Multimodal)")
    parser.add_argument("--repo_id", type=str, required=True, help="HF repository ID (e.g., 'user/dataset-name')")
    parser.add_argument("--dataset_root", type=Path, required=True, help="Root path of the local dataset")
    parser.add_argument(
        "--folders",
        type=str,
        nargs="+",
        required=True,
        help="Subfolders to upload (e.g., 'audio_extracted_16k' 'fake_audio_dataset' or 'videos_extracted')"
    )
    parser.add_argument(
        "--metadata_files",
        type=str,
        nargs="*",
        default=[],
        help="List of metadata files to upload (e.g., 'fake_audio_metadata.csv' 'fake_audio_metadata.json')"
    )
    parser.add_argument(
        "--primary_metadata",
        type=str,
        default=None,
        help="Primary CSV to mirror as 'metadata.csv' during upload for HF web previews (auto-deleted after upload)"
    )
    parser.add_argument("--private", action="store_true", help="Set the repository visibility to private")

    args = parser.parse_args()

    upload_dataset(
        repo_id=args.repo_id,
        dataset_root=args.dataset_root,
        folders=args.folders,
        metadata_files=args.metadata_files,
        primary_metadata_csv=args.primary_metadata,
        private=args.private
    )


if __name__ == "__main__":
    main()