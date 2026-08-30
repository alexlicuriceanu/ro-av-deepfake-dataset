python upload_dataset_hf.py \
    --repo_id "alexlicuriceanu/ro-dia-deepfake-audio" \
    --dataset_root "/export/home/acs/stud/a/alicuriceanu/ro-av-deepfake/ro-av-deepfake-dataset" \
    --folders audio_extracted_16k fake_audio_dataset \
    --metadata_files fake_audio_metadata.csv fake_audio_metadata.parquet fake_audio_metadata.json \
    --primary_metadata fake_audio_metadata.csv