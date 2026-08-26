#!/usr/bin/env python3
"""Batch face detection across videos with CPU decoders and one GPU YOLO worker.

This is a standalone high-throughput detection stage. It writes one JSON timeline
per source video and does not modify or replace extract_simple_clips.py.
"""

import argparse
import concurrent.futures
import json
import logging
import queue
import shutil
import threading
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import torch
from tqdm import tqdm
from ultralytics import YOLO

import video_io


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}
LOGGER = logging.getLogger(__name__)


class DecodeProgress:
    """Allocate temporary tqdm rows to concurrently decoding videos."""

    def __init__(self, worker_count: int) -> None:
        self.available_positions = list(range(1, worker_count + 1))
        self.lock = threading.Lock()

    def create(self, video_path: Path, total: int) -> tqdm:
        with self.lock:
            position = self.available_positions.pop()
        return tqdm(
            total=total,
            desc=f"Decode {video_path.name[:30]}",
            unit="frame",
            position=position,
            leave=False,
            dynamic_ncols=True,
        )

    def close(self, progress: tqdm) -> None:
        position = progress.pos
        progress.close()
        with self.lock:
            self.available_positions.append(position)


@dataclass
class FrameChunk:
    video_path: Path
    source_fps: float
    frame_numbers: list[int]
    frames: list


@dataclass
class DecodeFinished:
    video_path: Path
    source_fps: float | None
    frame_count: int | None
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("../videos_raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("../detections"))
    parser.add_argument("--model", type=Path, default=Path("models/yolov8x-face.pt"))
    parser.add_argument("--device", default="0", help="Ultralytics device, e.g. 0 or cpu")
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--confidence", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--gpu-batch-size", type=int, default=512)
    parser.add_argument("--decode-workers", type=int, default=6)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--queue-size", type=int, default=24)
    parser.add_argument("--quantize", type=str, default=False)
    parser.add_argument("--log-file", type=Path, help="Write file-only stage-one logs here")
    return parser.parse_args()


def decode_video(
    video_path: Path,
    sample_fps: float,
    chunk_size: int,
    work_queue: queue.Queue[FrameChunk | DecodeFinished],
    decode_progress: DecodeProgress,
) -> tuple[Path, float, int]:
    """Decode one source on CPU and push sampled frame chunks to the GPU queue."""
    progress: tqdm | None = None
    try:
        _, frame_count, width, height, source_fps = video_io.video_info(video_path)
        LOGGER.info(
            "%s: decoder started (frames=%d, %dx%d, fps=%.3f)",
            video_path.name, frame_count, width, height, source_fps,
        )
        sample_interval = max(1, round(source_fps / sample_fps))
        frames: list = []
        frame_numbers: list[int] = []
        frame_number = 0
        progress = decode_progress.create(video_path, frame_count)
        for sample_index, frame in enumerate(
            video_io.frames(video_path, width, height, sample_interval=sample_interval)
        ):
            frame_number = sample_index * sample_interval
            frames.append(frame)
            frame_numbers.append(frame_number)
            if len(frames) == chunk_size:
                work_queue.put(FrameChunk(video_path, source_fps, frame_numbers, frames))
                frames, frame_numbers = [], []
            progress.update(sample_interval)
        if frames:
            work_queue.put(FrameChunk(video_path, source_fps, frame_numbers, frames))
        work_queue.put(DecodeFinished(video_path, source_fps, frame_count))
        LOGGER.info("%s: decoder completed", video_path.name)
    except BaseException as error:
        LOGGER.exception("%s: decoder failed", video_path.name)
        work_queue.put(DecodeFinished(video_path, None, None, str(error)))
        raise
    finally:
        if progress is not None:
            decode_progress.close(progress)
    return video_path, source_fps, frame_count


def flush_batch(
    model: YOLO,
    frames: list,
    owners: list[tuple[Path, float, int]],
    args: argparse.Namespace,
    detections: dict[Path, list[dict[str, object]]],
) -> None:
    """Run one cross-video GPU batch and append serialized boxes to each video timeline."""
    if not frames:
        return
    LOGGER.info("GPU inference started for %d sampled frames", len(frames))
    results = model.predict(
        frames,
        conf=args.confidence,
        device=args.device,
        imgsz=args.imgsz,
        quantize=args.quantize,
        verbose=False,
    )
    for result, (video_path, source_fps, frame_number) in zip(results, owners):
        boxes: list[dict[str, object]] = []
        if result.boxes is not None:
            coordinates = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            for box, confidence in zip(coordinates, confidences):
                boxes.append({
                    "bbox_xyxy": [round(float(value), 2) for value in box],
                    "confidence": round(float(confidence), 5),
                })
        detections[video_path].append({
            "frame": frame_number,
            "time_sec": round(frame_number / source_fps, 4),
            "boxes": boxes,
        })
    LOGGER.info("GPU inference completed for %d sampled frames", len(frames))


def write_detection_cache(
    video_path: Path,
    source_fps: float,
    frame_count: int,
    args: argparse.Namespace,
    detections: dict[Path, list[dict[str, object]]],
) -> None:
    video_output_dir = args.output_dir / video_path.stem
    video_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = video_output_dir / "detections.json"
    output_path.write_text(
        json.dumps(
            {
                "source_video": str(video_path),
                "source_fps": source_fps,
                "source_frame_count": frame_count,
                "sample_fps": args.sample_fps,
                "confidence_threshold": args.confidence,
                "frames": detections[video_path],
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("%s: wrote %s with %d sampled frames", video_path.name, output_path, len(detections[video_path]))


def main() -> None:
    args = parse_args()
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg and ffprobe are required but were not found on PATH")
    if args.sample_fps <= 0 or args.gpu_batch_size <= 0 or args.chunk_size <= 0:
        raise ValueError("--sample-fps, --gpu-batch-size, and --chunk-size must be positive")
    if args.decode_workers <= 0 or args.queue_size <= 0:
        raise ValueError("--decode-workers and --queue-size must be positive")
    if not args.model.exists():
        raise FileNotFoundError(f"Face model not found: {args.model}")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot see a GPU")

    videos = sorted(path for path in args.input_dir.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_file or args.output_dir / "stage1.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(threadName)s %(levelname)s %(message)s"))
    root_logger.addHandler(file_handler)
    LOGGER.info(
        "Starting %d videos; device=%s; decode_workers=%d; gpu_batch_size=%d",
        len(videos), args.device, args.decode_workers, args.gpu_batch_size,
    )
    print(
        f"[INIT] {len(videos)} videos; device={args.device}; "
        f"decode_workers={args.decode_workers}; gpu_batch_size={args.gpu_batch_size}"
    )

    model = YOLO(str(args.model))
    work_queue: queue.Queue[FrameChunk | DecodeFinished] = queue.Queue(maxsize=args.queue_size)
    decode_progress = DecodeProgress(args.decode_workers)
    detections: dict[Path, list[dict[str, object]]] = defaultdict(list)
    source_info: dict[Path, tuple[float, int]] = {}
    pending_frames: dict[Path, int] = defaultdict(int)
    finalized_videos: set[Path] = set()
    failures: list[tuple[Path, BaseException]] = []
    gpu_frames: list = []
    gpu_owners: list[tuple[Path, float, int]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.decode_workers) as executor:
        futures = {
            executor.submit(
                decode_video, path, args.sample_fps, args.chunk_size, work_queue, decode_progress
            ): path
            for path in videos
        }
        completed_decoders = 0
        with tqdm(total=len(videos), desc="Videos", unit="video", position=0, dynamic_ncols=True) as progress:
            def finalize_ready_videos() -> None:
                for video_path, (source_fps, frame_count) in source_info.items():
                    if video_path not in finalized_videos and pending_frames[video_path] == 0:
                        write_detection_cache(video_path, source_fps, frame_count, args, detections)
                        finalized_videos.add(video_path)
                        progress.update(1)

            with tqdm(
                desc="GPU face inference",
                unit="frame",
                position=args.decode_workers + 1,
                leave=False,
                dynamic_ncols=True,
            ) as gpu_progress:
                gpu_batches = 0
                gpu_progress.set_postfix_str(
                    f"queued 0/{args.gpu_batch_size}; completed 0"
                )
                while completed_decoders < len(videos):
                    item = work_queue.get()
                    if isinstance(item, DecodeFinished):
                        completed_decoders += 1
                        if item.error is None and item.source_fps is not None and item.frame_count is not None:
                            source_info[item.video_path] = (item.source_fps, item.frame_count)
                        else:
                            failures.append((item.video_path, RuntimeError(item.error or "decoder failed")))
                        finalize_ready_videos()
                        continue
                    chunk = item
                    pending_frames[chunk.video_path] += len(chunk.frames)
                    for frame, frame_number in zip(chunk.frames, chunk.frame_numbers):
                        gpu_frames.append(frame)
                        gpu_owners.append((chunk.video_path, chunk.source_fps, frame_number))
                        if len(gpu_frames) == args.gpu_batch_size:
                            gpu_progress.set_postfix_str(
                                f"batch {gpu_batches + 1} running; queued {len(gpu_frames)}/{args.gpu_batch_size}"
                            )
                            gpu_progress.refresh()
                            flush_batch(model, gpu_frames, gpu_owners, args, detections)
                            gpu_progress.update(len(gpu_frames))
                            for owner_path, _, _ in gpu_owners:
                                pending_frames[owner_path] -= 1
                            gpu_batches += 1
                            gpu_frames, gpu_owners = [], []
                            finalize_ready_videos()
                    gpu_progress.set_postfix_str(
                        f"queued {len(gpu_frames)}/{args.gpu_batch_size}; completed {gpu_batches}"
                    )
                if gpu_frames:
                    gpu_progress.set_postfix_str(
                        f"batch {gpu_batches + 1} running; queued {len(gpu_frames)}/{args.gpu_batch_size}"
                    )
                    gpu_progress.refresh()
                    flush_batch(model, gpu_frames, gpu_owners, args, detections)
                    gpu_progress.update(len(gpu_frames))
                    for owner_path, _, _ in gpu_owners:
                        pending_frames[owner_path] -= 1
                    gpu_batches += 1
                    gpu_frames, gpu_owners = [], []
                    finalize_ready_videos()

        for future, video_path in futures.items():
            try:
                future.result()
            except BaseException as error:
                if not any(path == video_path for path, _ in failures):
                    failures.append((video_path, error))
    for video_path, error in failures:
        LOGGER.error("Skipping %s: %s", video_path, error)
        print(f"[SKIP] {video_path}: {error}")
    LOGGER.info("Stage 1 completed: caches=%d, failed=%d", len(finalized_videos), len(failures))


if __name__ == "__main__":
    main()