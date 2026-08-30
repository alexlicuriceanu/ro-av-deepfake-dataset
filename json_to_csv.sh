#!/bin/bash

python json_to_csv.py \
    --json_path ./fake_audio_metadata.json \
    --out_csv ./fake_audio_metadata.csv \
    --out_parquet ./fake_audio_metadata.parquet