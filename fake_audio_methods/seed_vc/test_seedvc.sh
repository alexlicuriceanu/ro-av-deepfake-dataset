#!/bin/bash
set -e

SRC_AUDIO="/export/home/acs/stud/a/alicuriceanu/ro-av-deepfake/ro-av-deepfake-dataset/audio_extracted_16k/maramures_itw_-Whhys-_3qE_full/000_000028.600.wav"
TGT1="/export/home/acs/stud/a/alicuriceanu/ro-av-deepfake/ro-av-deepfake-dataset/audio_extracted_16k/maramures_studio_rNlA--EuG-Y_110s-390s/011_000123.800.wav"
TGT2="/export/home/acs/stud/a/alicuriceanu/ro-av-deepfake/ro-av-deepfake-dataset/audio_extracted_16k/maramures_studio_rNlA--EuG-Y_110s-390s/005_000023.600.wav"
TGT3="/export/home/acs/stud/a/alicuriceanu/ro-av-deepfake/ro-av-deepfake-dataset/audio_extracted_16k/maramures_studio_rNlA--EuG-Y_110s-390s/009_000045.800.wav"

COMBINED_TGT="combined_target.wav"
OUT_DIR="."

echo "[1/2] Merging 3 target donor clips into a single reference"
ffmpeg -y -i "$TGT1" -i "$TGT2" -i "$TGT3" -filter_complex "[0:a][1:a][2:a]concat=n=3:v=0:a=1" "$COMBINED_TGT"

echo "[2/2] Running Seed-VC"
python seed-vc/inference.py \
    --source "$SRC_AUDIO" \
    --target "$COMBINED_TGT" \
    --output "$OUT_DIR" \
    --diffusion-steps 25 \
    --length-adjust 1.0 \
    --inference-cfg-rate 0.7 \
    --f0-condition False \
    --auto-f0-adjust False \
    --semi-tone-shift 0 \
    --fp16 True

echo "Saved to $OUT_DIR"