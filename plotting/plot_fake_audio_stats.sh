#!/bin/bash

python plot_fake_audio_stats.py \
    --manifest /export/home/acs/stud/a/alicuriceanu/ro-av-deepfake/ro-av-deepfake-dataset/master_manifest.json --run-eda \
    --run-audio-analysis \
    --out_dir /export/home/acs/stud/a/alicuriceanu/ro-av-deepfake/ro-av-deepfake-dataset/audio_plots \
    --eda-plot-name "05_fake_audio_eda.png" \
    --audio-plot-name "06_fake_audio_metrics.png" \
    --metrics-csv-name "06_fake_audio_metrics.csv"