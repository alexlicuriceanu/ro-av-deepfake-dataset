import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Convert fake_audio_metadata.json to flat CSV/Parquet")
    parser.add_argument(
        "--json_path",
        type=Path,
        default=Path("./fake_audio_metadata.json"),
        help="Path to fake_audio_metadata.json"
    )
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=Path("./metadata.csv"),
        help="Path for destination CSV"
    )
    parser.add_argument(
        "--out_parquet",
        type=Path,
        default=None,
        help="Optional path to also export as Parquet"
    )
    args = parser.parse_args()

    if not args.json_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {args.json_path}")

    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for sample_id, item in data.items():
        splits = item.get("splits", {})
        sample_type = item.get("type", "unknown")
        faking_method = item.get("faking_method")

        rows.append({
            "sample_id": sample_id,
            "master_key": item.get("key"),
            "clip_id": item.get("clip_id"),
            "video_id": item.get("video_id"),
            "file_name": item.get("path"),
            "type": sample_type,
            "label": 0 if sample_type == "real" else 1,
            "faking_method": faking_method if faking_method else "none",
            "dialect": item.get("dialect"),
            "environment": item.get("environment"),
            "split_protocol_1_ood": splits.get("protocol_1_ood", "unassigned"),
            "split_protocol_2_fewshot": splits.get("protocol_2_fewshot", "unassigned")
        })

    df = pd.DataFrame(rows)

    # Save CSV
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[SUCCESS] Exported CSV to: {args.out_csv.resolve()} ({len(df):,} rows)")

    # Optional Parquet export
    if args.out_parquet:
        args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out_parquet, index=False)
        print(f"[SUCCESS] Exported Parquet to: {args.out_parquet.resolve()}")


if __name__ == "__main__":
    main()