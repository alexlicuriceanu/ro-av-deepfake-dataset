#!/usr/bin/env python3
"""Create speaking clips from cached parallel YOLO detections.

Uses the exact tracking, VAD, mouth-motion, scene-cut, clip, and annotation
functions from extract_simple_clips.py without loading or running YOLO again.
"""

import argparse
import json
import logging
from logging.handlers import QueueHandler, QueueListener
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from silero_vad import load_silero_vad
from tqdm import tqdm

import extract_simple_clips as heuristic


WORKER_ARGS: argparse.Namespace | None = None
WORKER_VAD_MODEL = None
WORKER_PROGRESS_POSITION: int | None = None
LOGGER = logging.getLogger(__name__)


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
    parser.add_argument("--log-file", type=Path, help="Write file-only progress logs here")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def initialize_worker(
    args: argparse.Namespace,
    log_queue: Any,
    progress_lock: Any,
    progress_positions: Any,
) -> None:
    global WORKER_ARGS, WORKER_VAD_MODEL, WORKER_PROGRESS_POSITION
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(QueueHandler(log_queue))
    tqdm.set_lock(progress_lock)
    WORKER_ARGS = args
    WORKER_VAD_MODEL = load_silero_vad(onnx=True)
    WORKER_PROGRESS_POSITION = progress_positions.get()
    LOGGER.info("Worker initialized")


def load_detection_sets(payload: dict[str, Any]) -> list[list[heuristic.Detection]]:
    detection_sets: list[list[heuristic.Detection]] = []
    for frame_data in payload["frames"]:
        frame = int(frame_data["frame"])
        timestamp = float(frame_data["time_sec"])
        detection_sets.append([
            heuristic.Detection(frame, timestamp, np.asarray(box["bbox_xyxy"]), float(box["confidence"]))
            for box in frame_data["boxes"]
        ])
    return detection_sets


def process_cache(detection_path: Path) -> int:
    if WORKER_ARGS is None or WORKER_VAD_MODEL is None or WORKER_PROGRESS_POSITION is None:
        raise RuntimeError("Post-processing worker was not initialized")
    args = WORKER_ARGS
    with tqdm(
        total=4,
        desc=f"Process {detection_path.parent.name[:30]}",
        unit="phase",
        position=WORKER_PROGRESS_POSITION,
        leave=False,
        dynamic_ncols=True,
    ) as progress:
        progress.set_postfix_str("loading cache")
        LOGGER.info("Loading detection cache: %s", detection_path)
        payload = json.loads(detection_path.read_text(encoding="utf-8"))
        video_path = Path(payload["source_video"])
        duration, _, _, frame_height = heuristic.video_info(video_path)
        detection_sets = load_detection_sets(payload)
        tracks = heuristic.build_tracks(detection_sets, max_gap_samples=2, min_iou=0.30)
        progress.update(1)
        progress.set_postfix_str("running VAD")
        LOGGER.info(
            "%s: duration=%.1fs, samples=%d, tracks=%d",
            video_path.name, duration, len(detection_sets), len(tracks),
        )
        speech = heuristic.vad_intervals(video_path, WORKER_VAD_MODEL, args.vad_threshold)
        progress.update(1)
        progress.set_postfix_str("measuring mouths")
        LOGGER.info("%s: VAD found %d speech intervals", video_path.name, len(speech))
        heuristic.assign_talking_labels(
            video_path, tracks, speech, args.mouth_motion_threshold, args.mouth_open_threshold,
            args.speech_gap_seconds, args.face_landmarker_model, args.sample_fps,
        )
        progress.update(1)
        progress.set_postfix_str("selecting clips")
        cuts = heuristic.scene_changes(video_path, args.scene_threshold)
        candidates = heuristic.candidate_windows(
            tracks, frame_height, speech, cuts,
            args.min_clip_seconds, args.max_clip_seconds, args.sample_fps, args.min_face_height,
            args.min_presence, args.min_speech, args.max_talking_gap_seconds,
        )[:args.max_clips_per_video]
        progress.update(1)
        LOGGER.info("%s: selected %d clips", video_path.name, len(candidates))
        if not candidates:
            diagnostics = heuristic.candidate_diagnostics(
                tracks, frame_height, speech, cuts, args.min_clip_seconds, args.max_clip_seconds,
                args.sample_fps, args.min_face_height, args.min_presence, args.min_speech,
                args.max_talking_gap_seconds,
            )
            LOGGER.info("%s: zero-clip diagnostics: %s", video_path.name, diagnostics)
        output_dir = args.output_dir / video_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        progress.reset(total=max(1, len(candidates)))
        progress.set_description(f"Encode {video_path.name[:31]}")
        for index, candidate in enumerate(candidates):
            start, clip_duration = candidate["start_sec"], candidate["duration_sec"]
            clip_path = output_dir / f"{index:03d}_{start:010.3f}.mp4"
            progress.set_postfix_str(f"clip {index + 1}/{len(candidates)}")
            if not args.dry_run:
                LOGGER.info(
                    "%s: encoding clip %d/%d (%.3fs to %.3fs)",
                    video_path.name, index + 1, len(candidates), start, start + clip_duration,
                )
                heuristic.extract_clip(video_path, clip_path, start, clip_duration, tracks, args.sample_fps, args.debug)
                heuristic.clip_annotations(clip_path, video_path, candidate["track_id"], start, clip_duration, tracks)
            progress.update(1)
        if not candidates:
            progress.set_postfix_str("no clips selected")
            progress.update(1)
        LOGGER.info("%s: completed", video_path.name)
        return len(candidates)


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be at least 1")
    if not args.face_landmarker_model.exists():
        raise FileNotFoundError(f"Face Landmarker model not found: {args.face_landmarker_model}")
    caches = sorted(args.detections_dir.rglob("detections.json"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    context = multiprocessing.get_context("spawn")
    log_path = args.log_file or args.output_dir / "process.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(processName)s %(levelname)s %(message)s"))
    log_queue = context.Queue()
    progress_lock = context.RLock()
    progress_positions = context.Queue()
    for position in range(1, args.workers + 1):
        progress_positions.put(position)
    tqdm.set_lock(progress_lock)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(QueueHandler(log_queue))
    listener = QueueListener(log_queue, file_handler)
    listener.start()
    clip_count = 0
    LOGGER.info("Starting %d caches with %d workers", len(caches), args.workers)
    try:
        with ProcessPoolExecutor(
            args.workers,
            mp_context=context,
            initializer=initialize_worker,
            initargs=(args, log_queue, progress_lock, progress_positions),
        ) as executor:
            futures = {executor.submit(process_cache, cache): cache for cache in caches}
            with tqdm(total=len(caches), desc="Videos", unit="video", position=0, dynamic_ncols=True) as progress:
                for future in as_completed(futures):
                    try:
                        clip_count += future.result()
                    except (RuntimeError, OSError, ValueError, json.JSONDecodeError):
                        LOGGER.exception("Skipping cache: %s", futures[future])
                    finally:
                        progress.update(1)
        LOGGER.info("Completed %d clips from %d caches", clip_count, len(caches))
    finally:
        listener.stop()


if __name__ == "__main__":
    main()