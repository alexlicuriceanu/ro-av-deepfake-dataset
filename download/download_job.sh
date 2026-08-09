#!/bin/bash
#SBATCH --job-name=ro_av_dataset
#SBATCH --partition=haswell
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=12:00:00             # Safely covers the overnight run
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Load the apptainer module (check your HPC docs, sometimes it is loaded by default)
# module load apptainer 

# Define the path to your container image (.sif file)
CONTAINER_IMAGE="./downloader.sif"

# Run the python script INSIDE the container
# By default, Apptainer automatically mounts your current working directory ($PWD), 
# so it will instantly see your config.json, CSVs, and python script.
apptainer exec $CONTAINER_IMAGE python download_videos.py
