#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

exec python extract_clips_from_detections.py \
    --log-file ./stage2.log \
    --detections-dir ../videos_detections \
    --output-dir /tmp/alicuriceanu/videos_extracted \
    --workers 16 \
    --max-clips-per-video 15 \
    --min-clip-seconds 5 \
    --max-clip-seconds 10 \
    --max-talking-gap-seconds 1.0 \
    --mouth-motion-threshold 0.004 \
    --mouth-open-threshold 0.08 \
    --min-presence 0.70 \
    --min-speech 0.35