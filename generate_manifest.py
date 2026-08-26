import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

# ================= Global Defaults & Modality Configurations =================
DEFAULT_VIDEOS_ROOT = Path("/export/home/acs/stud/a/alicuriceanu/ro-av-deepfake/ro-av-deepfake-dataset/videos_extracted")
DEFAULT_AUDIO_ROOT = Path("/export/home/acs/stud/a/alicuriceanu/ro-av-deepfake/ro-av-deepfake-dataset/audio_extracted_16k")
DEFAULT_MANIFEST_PATH = Path("./master_manifest.json")

# Audio Modality Hyperparameters
AUDIO_METHODS = ["openvoice", "seed_vc", "cosyvoice"]
AUDIO_NUM_DONORS = 3

# Dataset & Splitting Defaults
DEFAULT_OOD_DIALECTS = ["rep_moldova", "maramures"]
DEFAULT_ID_SPLIT_RATIOS = (0.70, 0.15, 0.15)  # Train, Val, Test-ID
DEFAULT_OOD_FEWSHOT_RATIOS = (0.70, 0.15, 0.15)  # Train, Val, Test-OOD for Protocol 2

SEED = 1337


def parse_folder_name(folder_name: str) -> Tuple[str, str]:
    """
    Extracts dialect and environment from video folder names.
    Handles composite names like 'rep_moldova_studio_...'.
    """
    parts = folder_name.split("_")
    if parts[0] == "rep" and parts[1] == "moldova":
        dialect = "rep_moldova"
        env = parts[2]
    else:
        dialect = parts[0]
        env = parts[1]
    return dialect, env


def step_init(videos_root: Path, audio_root: Path, manifest_path: Path) -> Dict:
    """
    Scans VIDEOS_ROOT and AUDIO_ROOT, generating globally unique keys per clip.
    """
    videos_root = videos_root.resolve()
    audio_root = audio_root.resolve()
    manifest = {}

    clip_files = sorted(videos_root.glob("*/*.mp4"))
    if not clip_files:
        raise FileNotFoundError(f"No .mp4 clips found in {videos_root}")

    for clip_path in clip_files:
        folder_name = clip_path.parent.name
        dialect, env = parse_folder_name(folder_name)
        clip_stem = clip_path.stem

        # Globally unique key preventing dictionary collisions
        unique_key = f"{folder_name}__{clip_stem}"
        audio_path = audio_root / folder_name / f"{clip_stem}.wav"

        manifest[unique_key] = {
            "key": unique_key,
            "clip_id": clip_stem,
            "video_path": str(clip_path.resolve()),
            "audio_path": str(audio_path.resolve()),
            "video_id": folder_name,
            "dialect": dialect,
            "environment": env,
        }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[INIT] Manifest initialized with {len(manifest)} unique clip records at: {manifest_path}")
    return manifest


def step_split(
    manifest_path: Path,
    ood_dialects: List[str],
    id_ratios: Tuple[float, float, float],
    ood_fewshot_ratios: Tuple[float, float, float],
    seed: int = 42,
) -> Dict:
    """
    Computes speaker-disjoint splits for Protocol 1 & Protocol 2 and attaches 'splits' field.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}. Run --init first.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    random.seed(seed)

    # 1. Group unique videos by (dialect, environment)
    video_to_meta = {}
    for entry in manifest.values():
        vid = entry["video_id"]
        if vid not in video_to_meta:
            video_to_meta[vid] = {
                "dialect": entry["dialect"],
                "environment": entry["environment"],
            }

    grouped_videos = {}
    for vid, meta in video_to_meta.items():
        key = (meta["dialect"], meta["environment"])
        grouped_videos.setdefault(key, []).append(vid)

    # 2. Assign splits at the video (speaker) level
    video_splits_p1 = {}
    video_splits_p2 = {}

    for (dialect, env), vids in grouped_videos.items():
        random.shuffle(vids)
        n = len(vids)

        if dialect in ood_dialects:
            # Protocol 1: 100% Out-of-Domain Test
            for v in vids:
                video_splits_p1[v] = f"test_ood_{dialect}"

            # Protocol 2: Few-Shot Adaptation
            n_train = int(ood_fewshot_ratios[0] * n)
            n_val = int(ood_fewshot_ratios[1] * n)
            for i, v in enumerate(vids):
                if i < n_train:
                    video_splits_p2[v] = "train"
                elif i < n_train + n_val:
                    video_splits_p2[v] = "val"
                else:
                    video_splits_p2[v] = f"test_ood_{dialect}"
        else:
            # In-Domain Dialects
            n_train = int(id_ratios[0] * n)
            n_val = int(id_ratios[1] * n)
            for i, v in enumerate(vids):
                split = "train" if i < n_train else ("val" if i < n_train + n_val else "test_id")
                video_splits_p1[v] = split
                video_splits_p2[v] = split

    # 3. Update manifest entries
    for entry in manifest.values():
        vid = entry["video_id"]
        entry["splits"] = {
            "protocol_1_ood": video_splits_p1[vid],
            "protocol_2_fewshot": video_splits_p2[vid],
        }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[SPLIT] Successfully assigned Protocol 1 & 2 splits to {len(manifest)} entries.")
    return manifest


def step_audio(
    manifest_path: Path,
    audio_methods: List[str],
    num_donors: int,
    seed: int = 42,
) -> Dict:
    """
    Assigns balanced audio synthesis methods and samples donor audio paths
    from a single donor speaker per clip, guaranteeing strict speaker-disjointness
    across BOTH Protocol 1 and Protocol 2.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}. Run --init and --split first.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    random.seed(seed)

    # 1. Group clips by composite split tuple (P1, P2) to prevent any cross-split leakage
    composite_groups = {}
    for entry in manifest.values():
        p1 = entry["splits"]["protocol_1_ood"]
        p2 = entry["splits"]["protocol_2_fewshot"]
        key = (p1, p2)
        composite_groups.setdefault(key, []).append(entry)

    # 2. Assign methods and sample donor audios from the SAME speaker
    method_idx = 0
    for split_key, entries in composite_groups.items():
        vid_to_audios = {}
        for e in entries:
            vid_to_audios.setdefault(e["video_id"], []).append(e["audio_path"])

        available_vids = list(vid_to_audios.keys())

        for entry in entries:
            assigned_method = audio_methods[method_idx % len(audio_methods)]
            method_idx += 1

            candidate_vids = [v for v in available_vids if v != entry["video_id"]]
            selected_donor_audios = []
            chosen_donor_vid = None

            if candidate_vids:
                # 1. Pick a single disjoint donor speaker
                chosen_donor_vid = random.choice(candidate_vids)
                donor_clips = vid_to_audios[chosen_donor_vid]

                # 2. Sample up to num_donors audio clips from this exact donor speaker
                k_samples = min(num_donors, len(donor_clips))
                selected_donor_audios = random.sample(donor_clips, k=k_samples)

            entry["audio_manipulation"] = {
                "assigned_method": assigned_method,
                "donor_video_id": chosen_donor_vid,
                "donor_audio_paths": selected_donor_audios,
            }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[AUDIO] Assigned audio methods and single-speaker donor sets to {len(manifest)} entries.")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Master Manifest Generator for Romanian Audio-Visual Deepfake Dataset")

    parser.add_argument("--init", action="store_true", help="Initialize manifest with unique clip records.")
    parser.add_argument("--split", action="store_true", help="Compute speaker-disjoint splits for Protocol 1 & 2.")
    parser.add_argument("--audio", action="store_true", help="Assign audio fake methods and donor audio paths.")
    parser.add_argument("--all", action="store_true", help="Run --init, --split, and --audio sequentially.")

    parser.add_argument("--videos_root", type=Path, default=DEFAULT_VIDEOS_ROOT, help="Path to extracted clips directory.")
    parser.add_argument("--audio_root", type=Path, default=DEFAULT_AUDIO_ROOT, help="Path to extracted 16kHz audio directory.")
    parser.add_argument("--manifest_path", type=Path, default=DEFAULT_MANIFEST_PATH, help="Path to output JSON manifest.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility.")

    parser.add_argument("--ood_dialects", nargs="+", default=DEFAULT_OOD_DIALECTS, help="List of out-of-domain dialect names.")
    parser.add_argument("--id_ratios", nargs=3, type=float, default=DEFAULT_ID_SPLIT_RATIOS, help="Train Val Test-ID ratios.")
    parser.add_argument("--ood_fewshot_ratios", nargs=3, type=float, default=DEFAULT_OOD_FEWSHOT_RATIOS, help="Train Val Test-OOD ratios for Protocol 2.")

    parser.add_argument("--audio_methods", nargs="+", default=AUDIO_METHODS, help="List of zero-shot audio faking methods.")
    parser.add_argument("--num_audio_donors", type=int, default=AUDIO_NUM_DONORS, help="Number of donor voice samples to assign.")

    args = parser.parse_args()

    if not (args.init or args.split or args.audio or args.all):
        parser.print_help()
        return

    if args.init or args.all:
        step_init(args.videos_root, args.audio_root, args.manifest_path)

    if args.split or args.all:
        step_split(
            manifest_path=args.manifest_path,
            ood_dialects=args.ood_dialects,
            id_ratios=tuple(args.id_ratios),
            ood_fewshot_ratios=tuple(args.ood_fewshot_ratios),
            seed=args.seed,
        )

    if args.audio or args.all:
        step_audio(
            manifest_path=args.manifest_path,
            audio_methods=args.audio_methods,
            num_donors=args.num_audio_donors,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()