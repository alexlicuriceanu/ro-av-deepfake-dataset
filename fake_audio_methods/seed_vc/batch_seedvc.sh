#!/bin/bash
set -e

MANIFEST_PATH="/export/home/acs/stud/a/alicuriceanu/ro-av-deepfake/ro-av-deepfake-dataset/master_manifest.json"
OUT_DIR="/export/home/acs/stud/a/alicuriceanu/ro-av-deepfake/ro-av-deepfake-dataset/fake_audio_dataset/seedvc"

python batch_seedvc.py \
    --manifest "$MANIFEST_PATH" \
    --out_dir "$OUT_DIR" \
    --diffusion-steps 25 \
    --inference-cfg-rate 0.7 \
    --length-adjust 1.0 \
    --fp16 True