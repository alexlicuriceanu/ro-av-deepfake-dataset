import argparse
import os
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple
from tqdm import tqdm

DEFAULT_VIDEOS_ROOT = Path("/export/home/acs/stud/a/alicuriceanu/ro-av-deepfake/ro-av-deepfake-dataset/videos_extracted")
DEFAULT_AUDIO_ROOT = Path("/export/home/acs/stud/a/alicuriceanu/ro-av-deepfake/ro-av-deepfake-dataset/audio_extracted_16k")


def extract_audio_from_clip(
    video_path: Path,
    out_audio_path: Path,
    sample_rate: int = 16000,
    overwrite: bool = False,
) -> Tuple[bool, Optional[str]]:
    """
    Extracts 16-bit mono PCM audio from video using FFmpeg.
    """
    if out_audio_path.exists() and not overwrite:
        if out_audio_path.stat().st_size > 0:
            return True, None

    out_audio_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        str(out_audio_path),
    ]

    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            return False, f"FFmpeg error on {video_path.name}: {res.stderr.strip()}"
        return True, None
    except Exception as e:
        return False, f"Exception on {video_path.name}: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description="Multithreaded 16kHz Audio Extraction from Video Clips")
    parser.add_argument(
        "--videos_root",
        type=Path,
        default=DEFAULT_VIDEOS_ROOT,
        help="Path to root directory containing extracted video subfolders",
    )
    parser.add_argument(
        "--audio_root",
        type=Path,
        default=DEFAULT_AUDIO_ROOT,
        help="Target folder for extracted .wav files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(32, (os.cpu_count() or 1) * 2),
        help="Number of parallel worker threads",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=16000,
        help="Target audio sampling rate (default: 16000)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .wav files",
    )
    parser.add_argument(
        "--preserve_structure",
        action="store_true",
        help="Preserve subfolder hierarchy (audio_root/video_id/clip_id.wav) instead of flat layout",
    )
    args = parser.parse_args()

    args.audio_root.mkdir(parents=True, exist_ok=True)
    video_files = sorted(args.videos_root.glob("*/*.mp4"))

    if not video_files:
        print(f"[ERROR] No .mp4 clips found under {args.videos_root.resolve()}")
        return

    print(f"Found {len(video_files)} video clips to process.")
    print(f"Destination: {args.audio_root.resolve()}")
    print(f"Worker threads: {args.workers} | Target SR: {args.sr} Hz")

    tasks = []
    for video_path in video_files:
        if args.preserve_structure:
            dest_wav = args.audio_root / video_path.parent.name / f"{video_path.stem}.wav"
        else:
            dest_wav = args.audio_root / f"{video_path.stem}.wav"
        tasks.append((video_path, dest_wav))

    success_count = 0
    fail_count = 0
    errors = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                extract_audio_from_clip,
                v_path,
                a_path,
                args.sr,
                args.overwrite,
            ): (v_path, a_path)
            for v_path, a_path in tasks
        }

        with tqdm(total=len(futures), desc="Extracting Audio", unit="clip") as pbar:
            for future in as_completed(futures):
                success, err_msg = future.result()
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                    if err_msg:
                        errors.append(err_msg)
                pbar.update(1)

    print("\n================ Extraction Summary ================")
    print(f"Total processed : {len(tasks)}")
    print(f"Successful      : {success_count}")
    print(f"Failed          : {fail_count}")

    if errors:
        print("\nFailures:")
        for err in errors[:10]:
            print(f"  - {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors.")
    print("=====================================================")


if __name__ == "__main__":
    main()