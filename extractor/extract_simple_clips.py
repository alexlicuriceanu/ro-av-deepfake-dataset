#!/usr/bin/env python3
"""Extract single-face, speech-active clips without an active-speaker model.

The script samples video frames, batches face detection on the GPU, builds simple
IoU face tracks, and retains fixed-duration windows with one dominant face, no
detected cut, and enough neural-VAD speech. It is deliberately a conservative
candidate generator: VAD establishes that someone is speaking, not that the
visible person is the speaker.
"""

import argparse
import bisect
import contextlib
import csv
import json
import multiprocessing
import os
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, TypedDict

import cv2
os.environ.setdefault("GLOG_minloglevel", "2")
import mediapipe as mp
import numpy as np
import torch
import soundfile as sf
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from silero_vad import get_speech_timestamps, load_silero_vad
from tqdm import tqdm
from ultralytics import YOLO

import video_io

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}

WORKER_MODEL: YOLO | None = None
WORKER_VAD_MODEL: torch.nn.Module | None = None
WORKER_ARGS: argparse.Namespace | None = None


@contextlib.contextmanager
def suppress_native_stderr() -> Iterator[None]:
    """Silence MediaPipe's native startup and shutdown logs."""
    saved_stderr = os.dup(2)
    try:
        with open(os.devnull, "w", encoding="utf-8") as null_device:
            os.dup2(null_device.fileno(), 2)
            yield
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)


@dataclass
class Detection:
    frame: int
    time: float
    box: np.ndarray
    confidence: float
    mouth_motion: float = 0.0
    is_talking: bool = False


class Candidate(TypedDict):
    track_id: int
    start_sec: float
    end_sec: float
    duration_sec: float
    presence_ratio: float
    speech_ratio: float
    talking_ratio: float
    face_height_ratio: float
    mean_confidence: float
    median_bbox: list[float]


def run(command: list[str], capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def video_info(path: Path) -> tuple[float, int, int, int]:
    duration, frame_count, width, height, _ = video_io.video_info(path)
    return duration, frame_count, width, height


def iou(first: np.ndarray, second: np.ndarray) -> float:
    left_top = np.maximum(first[:2], second[:2])
    right_bottom = np.minimum(first[2:], second[2:])
    overlap = np.maximum(0.0, right_bottom - left_top)
    intersection = overlap[0] * overlap[1]
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return float(intersection / union) if union else 0.0


def batched_detections(
    video_path: Path,
    model: YOLO,
    sample_fps: float,
    batch_size: int,
    confidence: float,
    device: str,
) -> tuple[list[list[Detection]], list[float]]:
    """Decode only sampled frames and submit them to YOLO in GPU-sized batches."""
    _, frame_count, width, height, source_fps = video_io.video_info(video_path)
    sample_interval = max(1, round(source_fps / sample_fps))
    detection_sets: list[list[Detection]] = []
    timestamps: list[float] = []
    images: list[np.ndarray] = []
    frame_numbers: list[int] = []

    def infer_batch() -> None:
        if not images:
            return
        results = model.predict(
            images,
            conf=confidence,
            device=device,
            verbose=False,
        )
        for result, frame_number in zip(results, frame_numbers):
            frame_detections: list[Detection] = []
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                for box, score in zip(boxes, scores):
                    frame_detections.append(
                        Detection(frame_number, frame_number / source_fps, box, float(score))
                    )
            detection_sets.append(frame_detections)
            timestamps.append(frame_number / source_fps)
        images.clear()
        frame_numbers.clear()

    frame_number = 0
    with tqdm(total=frame_count, desc="  Detecting faces", unit="frame", disable=True) as progress:
        for sample_index, image in enumerate(
            video_io.frames(video_path, width, height, sample_interval=sample_interval)
        ):
            frame_number = sample_index * sample_interval
            images.append(image)
            frame_numbers.append(frame_number)
            if len(images) == batch_size:
                infer_batch()
            progress.update(sample_interval)
    infer_batch()
    return detection_sets, timestamps


def build_tracks(
    detection_sets: list[list[Detection]],
    max_gap_samples: int,
    min_iou: float,
) -> dict[int, list[Detection]]:
    tracks: dict[int, list[Detection]] = defaultdict(list)
    next_track_id = 0
    active: dict[int, tuple[int, Detection]] = {}

    for sample_index, detections in enumerate(detection_sets):
        active = {
            track_id: state
            for track_id, state in active.items()
            if sample_index - state[0] <= max_gap_samples
        }
        unmatched_tracks = set(active)
        for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
            candidates = [
                (iou(active[track_id][1].box, detection.box), track_id)
                for track_id in unmatched_tracks
            ]
            best_iou, track_id = max(candidates, default=(0.0, -1))
            if best_iou < min_iou:
                track_id = next_track_id
                next_track_id += 1
            else:
                unmatched_tracks.remove(track_id)
            tracks[track_id].append(detection)
            active[track_id] = (sample_index, detection)
    return tracks


def vad_intervals(video_path: Path, model: torch.nn.Module, threshold: float) -> list[tuple[float, float]]:
    """Return 16 kHz Silero VAD speech intervals; this is audio-only evidence."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        audio_path = Path(temporary_directory) / "audio.wav"
        run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(audio_path),
        ])
        samples, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    if sample_rate != 16000:
        raise RuntimeError(f"Expected 16 kHz WAV from FFmpeg, got {sample_rate} Hz")
    audio = torch.from_numpy(np.asarray(samples, dtype=np.float32))
    timestamps = get_speech_timestamps(
        audio,
        model,
        sampling_rate=16000,
        threshold=threshold,
        min_speech_duration_ms=250,
        min_silence_duration_ms=250,
    )
    return [(item["start"] / 16000, item["end"] / 16000) for item in timestamps]


def interval_coverage(intervals: Iterable[tuple[float, float]], start: float, end: float) -> float:
    covered = sum(max(0.0, min(end, right) - max(start, left)) for left, right in intervals)
    return covered / (end - start)


def is_speech_at_time(
    intervals: list[tuple[float, float]], timestamp: float, tolerance_seconds: float
) -> bool:
    """Treat short gaps around VAD intervals as speech to avoid label flicker."""
    return any(
        start - tolerance_seconds <= timestamp <= end + tolerance_seconds
        for start, end in intervals
    )


def mouth_aspect_ratio(
    frame: np.ndarray, box: np.ndarray, face_landmarker: vision.FaceLandmarker
) -> float | None:
    """Measure normalized lip opening on a detected face crop using Face Landmarker."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = np.clip(np.rint(box).astype(int), [0, 0, 0, 0], [width, height, width, height])
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
    )
    result = face_landmarker.detect(image)
    if not result.face_landmarks:
        return None
    landmarks = result.face_landmarks[0]
    vertical = abs(landmarks[13].y - landmarks[14].y)
    horizontal = abs(landmarks[78].x - landmarks[308].x)
    return vertical / max(horizontal, 1e-6)


def assign_talking_labels(
    video_path: Path,
    tracks: dict[int, list[Detection]],
    speech: list[tuple[float, float]],
    motion_threshold: float,
    open_threshold: float,
    speech_gap_seconds: float,
    face_landmarker_model: Path,
    sample_fps: float,
) -> None:
    """Set per-face speaking labels from VAD and temporal mouth-opening change.

    Audio VAD gates the decision; mouth motion attributes that speech to a visible
    track. This is a heuristic, not a replacement for a trained ASD model.
    """
    _, frame_count, width, height, source_fps = video_io.video_info(video_path)
    sample_interval = max(1, round(source_fps / sample_fps))
    by_frame: dict[int, list[tuple[int, Detection]]] = defaultdict(list)
    for track_id, detections in tracks.items():
        for detection in detections:
            by_frame[detection.frame].append((track_id, detection))
    previous_opening: dict[int, float] = {}
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(face_landmarker_model)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    with suppress_native_stderr():
        face_landmarker = vision.FaceLandmarker.create_from_options(options)
    try:
        with tqdm(total=frame_count, desc="  Measuring mouths", unit="frame", disable=True) as progress:
            for sample_index, frame in enumerate(
                video_io.frames(video_path, width, height, sample_interval=sample_interval)
            ):
                frame_number = sample_index * sample_interval
                for track_id, detection in by_frame.get(frame_number, []):
                    opening = mouth_aspect_ratio(frame, detection.box, face_landmarker)
                    if opening is None:
                        continue
                    track_previous = previous_opening.get(track_id)
                    detection.mouth_motion = (
                        abs(opening - track_previous) if track_previous is not None else 0.0
                    )
                    previous_opening[track_id] = opening
                    detection.is_talking = is_speech_at_time(
                        speech, detection.time, speech_gap_seconds
                    ) and (
                        detection.mouth_motion >= motion_threshold or opening >= open_threshold
                    )
                progress.update(sample_interval)
    finally:
        with suppress_native_stderr():
            face_landmarker.close()


def scene_changes(video_path: Path, threshold: float) -> list[float]:
    command = [
        "ffmpeg", "-hide_banner", "-i", str(video_path), "-filter:v",
        f"select='gt(scene,{threshold})',showinfo", "-an", "-f", "null", "-",
    ]
    completed = subprocess.run(command, text=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return [float(value) for value in re.findall(r"pts_time:([0-9.]+)", completed.stderr)]


def speaking_runs(
    detections: list[Detection], max_gap_seconds: float
) -> list[list[Detection]]:
    """Group positive samples while tolerating short negative-label gaps."""
    runs: list[list[Detection]] = []
    current: list[Detection] = []
    for detection in detections:
        if not detection.is_talking:
            continue
        if current and detection.time - current[-1].time > max_gap_seconds:
            runs.append(current)
            current = []
        current.append(detection)
    if current:
        runs.append(current)
    return runs


def candidate_windows(
    tracks: dict[int, list[Detection]],
    frame_height: int,
    speech: list[tuple[float, float]],
    cuts: list[float],
    min_clip_seconds: float,
    max_clip_seconds: float,
    sample_fps: float,
    min_face_height: float,
    min_presence: float,
    min_speech: float,
    max_talking_gap_seconds: float,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for track_id, detections in tracks.items():
        for run_detections in speaking_runs(detections, max_talking_gap_seconds):
            run_start = run_detections[0].time
            run_end = run_detections[-1].time + 1.0 / sample_fps
            start = run_start
            while start + min_clip_seconds <= run_end:
                end = min(start + max_clip_seconds, run_end)
                clip_duration = end - start
                expected_samples = max(1, round(clip_duration * sample_fps))
                in_window = [item for item in detections if start <= item.time < end]
                presence = len(in_window) / expected_samples
                if presence < min_presence or any(start < cut < end for cut in cuts):
                    start = end
                    continue
                boxes = np.stack([item.box for item in in_window])
                face_height = float(np.median(boxes[:, 3] - boxes[:, 1]) / frame_height)
                speech_ratio = interval_coverage(speech, start, end)
                talking_ratio = sum(item.is_talking for item in in_window) / len(in_window)
                if face_height < min_face_height or speech_ratio < min_speech:
                    start = end
                    continue
                confidences = [item.confidence for item in in_window]
                candidates.append({
                    "track_id": track_id,
                    "start_sec": round(start, 3),
                    "end_sec": round(end, 3),
                    "duration_sec": round(clip_duration, 3),
                    "presence_ratio": round(presence, 4),
                    "speech_ratio": round(speech_ratio, 4),
                    "talking_ratio": round(talking_ratio, 4),
                    "face_height_ratio": round(face_height, 4),
                    "mean_confidence": round(float(np.mean(confidences)), 4),
                    "median_bbox": [round(float(value), 1) for value in np.median(boxes, axis=0)],
                })
                start = end
    candidates.sort(
        key=lambda item: (
            item["talking_ratio"],
            item["speech_ratio"],
            item["face_height_ratio"],
            item["presence_ratio"],
        ),
        reverse=True,
    )
    selected: list[Candidate] = []
    for candidate in candidates:
        if not any(
            candidate["track_id"] == kept["track_id"]
            and candidate["start_sec"] < kept["end_sec"]
            and kept["start_sec"] < candidate["end_sec"]
            for kept in selected
        ):
            selected.append(candidate)
    return selected


def candidate_diagnostics(
    tracks: dict[int, list[Detection]],
    frame_height: int,
    speech: list[tuple[float, float]],
    cuts: list[float],
    min_clip_seconds: float,
    max_clip_seconds: float,
    sample_fps: float,
    min_face_height: float,
    min_presence: float,
    min_speech: float,
    max_talking_gap_seconds: float,
) -> dict[str, int | float]:
    """Summarize why the current candidate thresholds accepted no clip."""
    total_samples = sum(len(detections) for detections in tracks.values())
    talking_samples = sum(
        detection.is_talking for detections in tracks.values() for detection in detections
    )
    longest_run_seconds = 0.0
    windows = presence_rejected = cut_rejected = face_rejected = speech_rejected = 0
    for detections in tracks.values():
        for run_detections in speaking_runs(detections, max_talking_gap_seconds):
            run_start = run_detections[0].time
            run_end = run_detections[-1].time + 1.0 / sample_fps
            longest_run_seconds = max(longest_run_seconds, run_end - run_start)
            start = run_start
            while start + min_clip_seconds <= run_end:
                windows += 1
                end = min(start + max_clip_seconds, run_end)
                expected_samples = max(1, round((end - start) * sample_fps))
                in_window = [item for item in detections if start <= item.time < end]
                presence = len(in_window) / expected_samples
                if presence < min_presence:
                    presence_rejected += 1
                elif any(start < cut < end for cut in cuts):
                    cut_rejected += 1
                else:
                    boxes = np.stack([item.box for item in in_window])
                    face_height = float(np.median(boxes[:, 3] - boxes[:, 1]) / frame_height)
                    speech_ratio = interval_coverage(speech, start, end)
                    if face_height < min_face_height:
                        face_rejected += 1
                    elif speech_ratio < min_speech:
                        speech_rejected += 1
                start = end
    return {
        "total_samples": total_samples,
        "talking_samples": talking_samples,
        "longest_run_seconds": round(longest_run_seconds, 2),
        "eligible_talking_windows": windows,
        "presence_rejected": presence_rejected,
        "cut_rejected": cut_rejected,
        "face_rejected": face_rejected,
        "speech_rejected": speech_rejected,
    }


def nearest_detection(detections: list[Detection], frame_number: int, max_gap_frames: int) -> Detection | None:
    """Return a nearby sampled detection so boxes remain visible between samples."""
    frame_numbers = [item.frame for item in detections]
    position = bisect.bisect_left(frame_numbers, frame_number)
    candidates = detections[max(0, position - 1):position + 1]
    if not candidates:
        return None
    closest = min(candidates, key=lambda item: abs(item.frame - frame_number))
    return closest if abs(closest.frame - frame_number) <= max_gap_frames else None


def draw_track_overlays(
    frame: np.ndarray,
    tracks: dict[int, list[Detection]],
    frame_number: int,
    max_gap_frames: int,
) -> None:
    for track_id, detections in tracks.items():
        detection = nearest_detection(detections, frame_number, max_gap_frames)
        if detection is None:
            continue
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = np.clip(np.rint(detection.box).astype(int), [0, 0, 0, 0], [width - 1, height - 1, width - 1, height - 1])
        state = "TALKING" if detection.is_talking else "NOT TALKING"
        color = (40, 200, 40) if detection.is_talking else (40, 40, 220)
        label = f"ID {track_id}  {state}  {detection.confidence:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_top = max(0, y1 - label_size[1] - baseline - 6)
        cv2.rectangle(frame, (x1, label_top), (min(width - 1, x1 + label_size[0] + 6), y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - baseline - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


def extract_clip(
    video_path: Path,
    destination: Path,
    start: float,
    duration: float,
    tracks: dict[int, list[Detection]],
    sample_fps: float,
    debug: bool,
) -> None:
    """Encode the selected interval with per-frame face boxes and synchronized audio."""
    _, _, width, height, source_fps = video_io.video_info(video_path)
    first_frame = round(start * source_fps)
    frame_count = round(duration * source_fps)
    encoder_command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{width}x{height}",
        "-framerate", f"{source_fps:.6f}", "-i", "-", "-ss", f"{start:.3f}",
        "-t", f"{duration:.3f}", "-i", str(video_path), "-map", "0:v:0", "-map", "1:a?",
        "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p",
        "-r", "25", "-c:a", "copy",
        "-movflags", "+faststart", "-shortest", str(destination),
    ]
    encoder = subprocess.Popen(encoder_command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    pipe_closed = False
    try:
        with tqdm(total=frame_count, desc="  Encoding clip", unit="frame", disable=True) as progress:
            for offset, frame in enumerate(video_io.frames(video_path, width, height, start, duration)):
                if debug:
                    draw_track_overlays(
                        frame, tracks, first_frame + offset, max(1, round(source_fps / sample_fps))
                    )
                if encoder.stdin is None:
                    raise RuntimeError("FFmpeg video pipe is unavailable")
                encoder.stdin.write(frame.tobytes())
                progress.update(1)
    except BrokenPipeError:
        pipe_closed = True
    finally:
        if encoder.stdin is not None:
            try:
                encoder.stdin.close()
            except BrokenPipeError:
                pipe_closed = True
    error_output = encoder.stderr.read().decode("utf-8", errors="replace") if encoder.stderr else ""
    return_code = encoder.wait()
    if pipe_closed or return_code != 0:
        details = error_output.strip() or "FFmpeg produced no stderr output."
        raise RuntimeError(
            f"FFmpeg overlay encoding failed (exit code {return_code}): {details}\n"
            f"Command: {' '.join(encoder_command)}"
        )


def clip_annotations(
    clip_path: Path,
    video_path: Path,
    track_id: int,
    start: float,
    duration: float,
    tracks: dict[int, list[Detection]],
) -> None:
    """Write sampled speaker boxes in source-video coordinates for one extracted clip."""
    end = start + duration
    boxes = [
        {
            "time_sec": round(detection.time - start, 3),
            "source_time_sec": round(detection.time, 3),
            "frame": detection.frame,
            "bbox_xyxy": [round(float(value), 1) for value in detection.box],
            "confidence": round(detection.confidence, 4),
            "is_talking": detection.is_talking,
        }
        for detection in tracks[track_id]
        if start <= detection.time < end
    ]
    annotation_path = clip_path.with_suffix(".json")
    annotation_path.write_text(
        json.dumps(
            {
                "clip_filename": clip_path.name,
                "source_video": str(video_path),
                "speaker_track_id": track_id,
                "start_sec": round(start, 3),
                "duration_sec": round(duration, 3),
                "boxes": boxes,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("../videos_raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("../extracted_clips_simple"))
    parser.add_argument("--model", type=Path, default=Path("models/yolov8x-face.pt"))
    parser.add_argument("--face-landmarker-model", type=Path, default=Path("models/face_landmarker.task"))
    parser.add_argument("--device", default="0", help="Ultralytics device, e.g. 0 or cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--min-clip-seconds", type=float, default=5.0)
    parser.add_argument("--max-clip-seconds", type=float, default=10.0)
    parser.add_argument("--confidence", type=float, default=0.45)
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
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--debug", action="store_true", help="Draw face boxes and speaking labels on output clips")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def initialize_worker(args: argparse.Namespace) -> None:
    """Create isolated model instances for one spawned video-processing worker."""
    global WORKER_ARGS, WORKER_MODEL, WORKER_VAD_MODEL
    WORKER_ARGS = args
    WORKER_MODEL = YOLO(str(args.model))
    WORKER_VAD_MODEL = load_silero_vad(onnx=True)


def process_video(video_path: Path) -> list[dict[str, object]]:
    """Process one source video using this worker's persistent model instances."""
    if WORKER_ARGS is None or WORKER_MODEL is None or WORKER_VAD_MODEL is None:
        raise RuntimeError("Video worker was not initialized")
    args = WORKER_ARGS
    duration, _, _, frame_height = video_info(video_path)
    if duration < args.min_clip_seconds:
        return []
    print(f"[VIDEO] {video_path.name} ({duration:.1f}s)")
    detection_sets, _ = batched_detections(
        video_path, WORKER_MODEL, args.sample_fps, args.batch_size, args.confidence, args.device
    )
    tracks = build_tracks(detection_sets, max_gap_samples=2, min_iou=0.30)
    speech = vad_intervals(video_path, WORKER_VAD_MODEL, args.vad_threshold)
    assign_talking_labels(
        video_path,
        tracks,
        speech,
        args.mouth_motion_threshold,
        args.mouth_open_threshold,
        args.speech_gap_seconds,
        args.face_landmarker_model,
        args.sample_fps,
    )
    cuts = scene_changes(video_path, args.scene_threshold)
    candidates = candidate_windows(
        tracks,
        frame_height,
        speech,
        cuts,
        args.min_clip_seconds,
        args.max_clip_seconds,
        args.sample_fps,
        args.min_face_height,
        args.min_presence,
        args.min_speech,
        args.max_talking_gap_seconds,
    )[:args.max_clips_per_video]
    print(f"  tracks={len(tracks)} candidates={len(candidates)}")
    video_output_dir = args.output_dir / video_path.stem
    video_output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for index, candidate in enumerate(candidates):
        clip_name = f"{index:03d}_{float(candidate['start_sec']):010.3f}.mp4"
        output_path = video_output_dir / clip_name
        start = float(candidate["start_sec"])
        clip_duration = float(candidate["duration_sec"])
        if not args.dry_run:
            extract_clip(video_path, output_path, start, clip_duration, tracks, args.sample_fps, args.debug)
            clip_annotations(output_path, video_path, int(candidate["track_id"]), start, clip_duration, tracks)
        records.append({
            "clip_filename": str(output_path.relative_to(args.output_dir)),
            "source_video": str(video_path),
            **candidate,
            "median_bbox": json.dumps(candidate["median_bbox"]),
        })
    return records


def main() -> None:
    args = parse_args()
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg and ffprobe are required but were not found on PATH")
    if not args.model.exists():
        raise FileNotFoundError(f"Face model not found: {args.model}")
    if not args.face_landmarker_model.exists():
        raise FileNotFoundError(
            f"Face Landmarker model not found: {args.face_landmarker_model}. "
            "Download the MediaPipe FaceLandmarker task bundle and pass its path with "
            "--face-landmarker-model."
        )
    if args.min_clip_seconds <= 0 or args.max_clip_seconds < args.min_clip_seconds:
        raise ValueError("--max-clip-seconds must be greater than or equal to --min-clip-seconds")
    if args.workers <= 0:
        raise ValueError("--workers must be at least 1")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot see a GPU")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_dir / "metadata.csv"
    videos = sorted(path for path in args.input_dir.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS)
    print(f"[INIT] {len(videos)} videos; device={args.device}; workers={args.workers}; batch_size={args.batch_size}")
    records: list[dict[str, object]] = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        initializer=initialize_worker,
        initargs=(args,),
    ) as executor:
        futures = {executor.submit(process_video, video_path): video_path for video_path in videos}
        with tqdm(total=len(videos), desc="Videos", unit="video") as progress:
            for future in as_completed(futures):
                video_path = futures[future]
                try:
                    records.extend(future.result())
                except (RuntimeError, subprocess.CalledProcessError, cv2.error) as error:
                    print(f"[SKIP] {video_path}: {error}")
                finally:
                    progress.update(1)

    fieldnames = [
        "clip_filename", "source_video", "duration_sec", "track_id", "start_sec", "end_sec",
        "presence_ratio", "speech_ratio", "talking_ratio", "face_height_ratio", "mean_confidence",
        "median_bbox",
    ]
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"[DONE] Wrote {len(records)} clips and {metadata_path}")


if __name__ == "__main__":
    main()