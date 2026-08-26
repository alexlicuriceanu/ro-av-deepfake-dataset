import csv
import json
import os
import sys
import logging
from urllib.parse import urlparse, parse_qs
import yt_dlp
from yt_dlp.utils import download_range_func
from tqdm import tqdm

def setup_logger(log_file):
    """Sets up a file-only logger for yt-dlp."""
    logger = logging.getLogger("yt_dlp_logger")
    logger.setLevel(logging.DEBUG)
    
    # Prevent logger from propagating to the console logger
    logger.propagate = False
    
    if logger.hasHandlers():
        logger.handlers.clear()
        
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    return logger

def parse_timestamp_to_seconds(ts_str):
    if not ts_str:
        return None
    parts = ts_str.strip().split(':')
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part)
    return seconds

def extract_video_id(url):
    parsed_url = urlparse(url)
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            p = parse_qs(parsed_url.query)
            return p.get('v', [None])[0]
    return "unknown_id"

def process_csv(csv_path, dialect_fallback, config, file_logger):
    tasks = []
    
    try:
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                tasks.append(row)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return

    output_dir = config.get("output_dir", "dataset_raw")
    os.makedirs(output_dir, exist_ok=True)
    
    pbar = tqdm(tasks, desc=f"Processing {os.path.basename(csv_path)}", unit="file", position=0, leave=True)

    for row in pbar:
        url = row.get("Link", "").strip()
        if not url:
            continue

        # Prioritize Dialect column from CSV if present, otherwise use config value
        dialect = row.get("Dialect", dialect_fallback).replace(" ", "")
        category = row.get("Category", "Unknown").replace(" ", "")
        trim_str = row.get("Trim", "").strip()
        vid_id = extract_video_id(url)

        if not trim_str:
            start_time, end_time = None, None
            file_suffix = "full"
            trim_log = "FULL VIDEO"
        else:
            try:
                start_str, end_str = trim_str.split('-')
                start_time = parse_timestamp_to_seconds(start_str)
                end_time = parse_timestamp_to_seconds(end_str)
                file_suffix = f"{int(start_time)}s-{int(end_time)}s"
                trim_log = f"{trim_str} ({int(start_time)}s-{int(end_time)}s)"
            except Exception:
                file_logger.error(f"Invalid Trim format for {vid_id}: '{trim_str}'. Expected MM:SS-MM:SS. Skipping.")
                continue

        # Output format: Dialect_Category_VideoID_Suffix.mp4
        out_filename = f"{dialect.lower()}_{category.lower()}_{vid_id}_{file_suffix}.mp4"
        out_path = os.path.join(output_dir, out_filename)

        if os.path.exists(out_path):
            file_logger.info(f"Already exists, skipping: {out_filename}")
            continue

        ydl_opts = config.get("yt_dlp_options", {}).copy()
        ydl_opts['outtmpl'] = out_path
        ydl_opts['logger'] = file_logger

        if start_time is not None and end_time is not None:
            ydl_opts['download_ranges'] = download_range_func(None, [(start_time, end_time)])

        file_logger.info(f"Starting: {vid_id} [{trim_log}]")
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            file_logger.info(f"[SUCCESS] Saved to {out_filename}")
        except Exception as e:
            file_logger.error(f"[FAILED] Error downloading {vid_id}: {e}")

def main():
    if not os.path.exists("config.json"):
        print("Error: config.json not found in current directory.")
        sys.exit(1)

    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    csv_entries = config.get("csv_files", [])
    if not csv_entries:
        print("No CSV files specified in config.json.")
        sys.exit(1)

    log_file_path = config.get("log_file", "extractor.log")
    file_logger = setup_logger(log_file_path)
    
    print(f"Logging yt-dlp output directly to: {log_file_path}")

    for entry in csv_entries:
        if isinstance(entry, dict):
            csv_path = entry.get("file")
            dialect = entry.get("dialect", "unknown")
        else:
            # Fallback for plain string entries
            csv_path = entry
            dialect = "unknown"

        if csv_path and os.path.exists(csv_path):
            process_csv(csv_path, dialect, config, file_logger)
        else:
            print(f"Warning: CSV file '{csv_path}' not found on disk. Skipping.")

if __name__ == "__main__":
    main()