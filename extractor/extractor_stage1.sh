#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

exec python extract_detections.py \
    --input-dir ../videos_raw \
    --output-dir ../videos_detections \
    --device 0 \
    --decode-workers 8 \
    --gpu-batch-size 512 \
    --chunk-size 64 \
    --queue-size 48 \
    --quantize "fp16" \
    --log-file ./stage1.log