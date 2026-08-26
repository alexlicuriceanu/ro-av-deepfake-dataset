#!/usr/bin/env python3
"""Create speaking clips from cached parallel YOLO detections.

Uses the exact tracking, VAD, mouth-motion, scene-cut, clip, and annotation
functions from extract_simple_clips.py without loading or running YOLO again.
"""

import argparse
import csv
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from silero_vad import load_silero_vad
from tqdm import tqdm

import extract_simple_clips as heuristic


WORKER_ARGS: argparse.Namespace | None = None
WORKER_VAD_MODEL = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections-dir", type=Path, default=Path("../detections"))
    parser.add_argument("--output-dir", type=Path, default=Path("../extracted_clips"))
    parser.add_argument("--face-landmarker-model", type=Path, default=Path("models/face_landmarker.task"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--min-clip-seconds", type=float, default=5.0)
    parser.add_argument("--max-clip-seconds", type=float, default=10.0)
    parser.add_argument("--min-face-height", type=float, default=0.20)
    parser.add_argument("--min-presence", type=float, default=0.85)
    parser.add_argument("--min-speech", type=float, default=0.60)
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--mouth-motion-threshold", type=float, default=0.008)
    parser.add_argument("--mouth-open-threshold", type=float, default=0.10)
    parser.add_argument("--speech-gap-seconds", type=float, default=0.35)
    parser.add_argument("--max-talking-gap-seconds", type=float, default=0.6)
    parser.add_argument("--scene-threshold", type=float, default=0.35)
    parser.add_argument("--max-clips-per-video", type=int, default=20)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def initialize_worker(args: argparse.Namespace) -> None:
    global WORKER_ARGS, WORKER_VAD_MODEL
    WORKER_ARGS = args
    WORKER_VAD_MODEL = load_silero_vad(onnx=True)


def load_detection_sets(payload: dict[str, object]) -> list[list[heuristic.Detection]]:
    detection_sets: list[list[heuristic.Detection]] = []
    for frame_data in payload["frames"]:
        frame = int(frame_data["frame"])
        timestamp = float(frame_data["time_sec"])
        detection_sets.append([
            heuristic.Detection(frame, timestamp, np.asarray(box["bbox_xyxy"]), float(box["confidence"]))
            for box in frame_data["boxes"]
        ])
    return detection_sets


def process_cache(detection_path: Path) -> list[dict[str, object]]:
    if WORKER_ARGS is None or WORKER_VAD_MODEL is None:
        raise RuntimeError("Post-processing worker was not initialized")
    args = WORKER_ARGS
    payload = json.loads(detection_path.read_text(encoding="utf-8"))
    video_path = Path(payload["source_video"])
    duration, _, _, frame_height = heuristic.video_info(video_path)
    detection_sets = load_detection_sets(payload)
    tracks = heuristic.build_tracks(detection_sets, max_gap_samples=2, min_iou=0.30)
    speech = heuristic.vad_intervals(video_path, WORKER_VAD_MODEL, args.vad_threshold)
    heuristic.assign_talking_labels(
        video_path, tracks, speech, args.mouth_motion_threshold, args.mouth_open_threshold,
        args.speech_gap_seconds, args.face_landmarker_model, args.sample_fps,
    )
    candidates = heuristic.candidate_windows(
        tracks, frame_height, speech, heuristic.scene_changes(video_path, args.scene_threshold),
        args.min_clip_seconds, args.max_clip_seconds, args.sample_fps, args.min_face_height,
        args.min_presence, args.min_speech, args.max_talking_gap_seconds,
    )[:args.max_clips_per_video]
    output_dir = args.output_dir / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for index, candidate in enumerate(candidates):
        start, clip_duration = candidate["start_sec"], candidate["duration_sec"]
        clip_path = output_dir / f"{index:03d}_{start:010.3f}.mp4"
        if not args.dry_run:
            heuristic.extract_clip(video_path, clip_path, start, clip_duration, tracks, args.sample_fps, args.debug)
            heuristic.clip_annotations(clip_path, video_path, candidate["track_id"], start, clip_duration, tracks)
        records.append({"clip_filename": str(clip_path.relative_to(args.output_dir)), "source_video": str(video_path), **candidate, "median_bbox": json.dumps(candidate["median_bbox"])})
    return records


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be at least 1")
    if not args.face_landmarker_model.exists():
        raise FileNotFoundError(f"Face Landmarker model not found: {args.face_landmarker_model}")
    caches = sorted(args.detections_dir.rglob("detections.json"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(args.workers, mp_context=context, initializer=initialize_worker, initargs=(args,)) as executor:
        futures = {executor.submit(process_cache, cache): cache for cache in caches}
        with tqdm(total=len(caches), desc="Videos", unit="video") as progress:
            for future in as_completed(futures):
                try:
                    records.extend(future.result())
                except RuntimeError as error:
                    print(f"[SKIP] {futures[future]}: {error}")
                finally:
                    progress.update(1)
    fields = ["clip_filename", "source_video", "duration_sec", "track_id", "start_sec", "end_sec", "presence_ratio", "speech_ratio", "talking_ratio", "face_height_ratio", "mean_confidence", "median_bbox"]
    with (args.output_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(f"[DONE] Wrote {len(records)} clips")


if __name__ == "__main__":
    main()