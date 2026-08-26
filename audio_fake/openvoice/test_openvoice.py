import argparse
from pathlib import Path
import shutil
import tempfile
from typing import List
import torch
from huggingface_hub import snapshot_download
from openvoice import se_extractor
from openvoice.api import ToneColorConverter


def setup_openvoice(device: str = "cuda") -> ToneColorConverter:
    print("[1/2] Downloading / verifying OpenVoice V2 checkpoints from Hugging Face...")
    ckpt_dir = Path(
        snapshot_download(
            repo_id="myshell-ai/OpenVoiceV2",
            allow_patterns=["converter/*"],
        )
    )
    converter_dir = ckpt_dir / "converter"
    config_path = converter_dir / "config.json"
    ckpt_path = converter_dir / "checkpoint.pth"

    print(f"[2/2] Initializing Tone Color Converter on {device}...")
    tone_color_converter = ToneColorConverter(str(config_path), device=device)
    tone_color_converter.load_ckpt(str(ckpt_path))
    return tone_color_converter


def convert_voice(
    src_path: Path,
    tgt_paths: List[Path],
    out_path: Path,
    converter: ToneColorConverter,
    tau: float = 0.03,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="openvoice_se_"))

    try:
        # 1. Extract source embedding from the single clip being converted
        print(f"Extracting source speaker embedding: {src_path.name}")
        src_se, _ = se_extractor.get_se(
            str(src_path), converter, target_dir=str(temp_dir), vad=True
        )

        # 2. Extract and average speaker embeddings across all target donor clips
        tgt_ses = []
        for path in tgt_paths:
            print(f"Extracting target/donor speaker embedding: {path.name}")
            se, _ = se_extractor.get_se(
                str(path), converter, target_dir=str(temp_dir), vad=True
            )
            if not isinstance(se, torch.Tensor):
                se = torch.tensor(se)
            tgt_ses.append(se)

        if len(tgt_ses) == 1:
            target_se = tgt_ses[0]
        else:
            print(f"Averaging {len(tgt_ses)} donor embeddings for stable target timbre...")
            target_se = torch.mean(torch.stack(tgt_ses), dim=0)

        # 3. Perform tone color conversion
        print(f"Converting audio timbre: {src_path.name} -> Target Speaker ({len(tgt_paths)} donor clips)...")
        converter.convert(
            audio_src_path=str(src_path),
            src_se=src_se,
            tgt_se=target_se,
            output_path=str(out_path),
            tau=tau,
            message=" ",
        )
        print(f"[SUCCESS] Converted audio saved to: {out_path.resolve()}")

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="OpenVoice V2 Multi-Target Donor Conversion")
    parser.add_argument("--src", type=Path, required=True, help="Path to primary source content .wav (to be faked)")
    parser.add_argument("--tgt1", type=Path, required=True, help="Path to primary donor timbre .wav")
    parser.add_argument("--tgt2", type=Path, default=None, help="Optional secondary donor .wav from same donor speaker")
    parser.add_argument("--tgt3", type=Path, default=None, help="Optional tertiary donor .wav from same donor speaker")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("./test_openvoice_out.wav"),
        help="Destination path for synthesized .wav",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.03,
        help="Sampling temperature for normalizing flow",
    )
    args = parser.parse_args()

    # Collect all available target donor clips
    tgt_paths = [args.tgt1]
    if args.tgt2 is not None and args.tgt2.exists():
        tgt_paths.append(args.tgt2)
    if args.tgt3 is not None and args.tgt3.exists():
        tgt_paths.append(args.tgt3)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using compute device: {device}")
    print(f"Target donor clips used: {[p.name for p in tgt_paths]}")

    converter = setup_openvoice(device=device)
    convert_voice(
        src_path=args.src,
        tgt_paths=tgt_paths,
        out_path=args.out,
        converter=converter,
        tau=args.tau,
    )


if __name__ == "__main__":
    main()