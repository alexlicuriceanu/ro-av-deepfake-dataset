"""FFmpeg-based video probing and software frame decoding."""

import json
import subprocess
from collections.abc import Iterator
from pathlib import Path

import numpy as np


def _parse_frame_rate(value: str) -> float:
    numerator, denominator = value.split("/", maxsplit=1)
    return float(numerator) / float(denominator) if float(denominator) else 0.0


def video_info(path: Path) -> tuple[float, int, int, int, float]:
    """Return duration, frame count, dimensions, and frame rate via FFprobe."""
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_frames,duration,avg_frame_rate,r_frame_rate:format=duration",
        "-of", "json", str(path),
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    streams = json.loads(completed.stdout).get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {path}")
    stream = streams[0]
    width = int(stream["width"])
    height = int(stream["height"])
    frame_rate = _parse_frame_rate(stream.get("avg_frame_rate", "0/0"))
    if frame_rate <= 0:
        frame_rate = _parse_frame_rate(stream.get("r_frame_rate", "25/1")) or 25.0
    duration = float(stream.get("duration") or json.loads(completed.stdout).get("format", {}).get("duration") or 0.0)
    frame_count_value = stream.get("nb_frames")
    frame_count = int(frame_count_value) if frame_count_value and frame_count_value != "N/A" else round(duration * frame_rate)
    return duration, frame_count, width, height, frame_rate


def frames(
    path: Path,
    width: int,
    height: int,
    start: float | None = None,
    duration: float | None = None,
    sample_interval: int = 1,
) -> Iterator[np.ndarray]:
    """Decode BGR frames with FFmpeg software decoding and yield selected frames."""
    if sample_interval <= 0:
        raise ValueError("sample_interval must be positive")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
    if start is not None:
        command.extend(["-ss", f"{start:.6f}"])
    command.extend(["-hwaccel", "none", "-i", str(path)])
    if duration is not None:
        command.extend(["-t", f"{duration:.6f}"])
    if sample_interval > 1:
        command.extend(["-vf", f"select=not(mod(n\\,{sample_interval}))"])
    command.extend(["-map", "0:v:0", "-an", "-sn", "-dn", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"])
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_size = width * height * 3
    try:
        if process.stdout is None:
            raise RuntimeError("FFmpeg frame pipe is unavailable")
        while True:
            data = process.stdout.read(frame_size)
            if not data:
                break
            if len(data) != frame_size:
                raise RuntimeError(f"FFmpeg returned an incomplete frame for {path}")
            yield np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
    finally:
        if process.stdout is not None:
            process.stdout.close()
    error_output = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"FFmpeg software decoding failed for {path}: {error_output.strip()}")