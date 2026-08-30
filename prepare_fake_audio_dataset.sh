#!/bin/bash

python prepare_fake_audio_dataset.py \
    --manifest /export/home/acs/stud/a/alicuriceanu/ro-av-deepfake/ro-av-deepfake-dataset/master_manifest.json \
    --dataset_root /export/home/acs/stud/a/alicuriceanu/ro-av-deepfake/ro-av-deepfake-dataset \
    --output /export/home/acs/stud/a/alicuriceanu/ro-av-deepfake/ro-av-deepfake-dataset/fake_audio_metadata.json \
    --check_files