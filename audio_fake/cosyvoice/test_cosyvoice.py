import argparse
import sys
import tempfile
from pathlib import Path
import torch
import torchaudio

CURRENT_DIR = Path(__file__).resolve().parent
REPO_DIR = CURRENT_DIR / "CosyVoice"

if not REPO_DIR.exists():
    raise FileNotFoundError(f"Could not find CosyVoice repo at {REPO_DIR}")

sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "third_party" / "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import CosyVoice3
from cosyvoice.utils.file_utils import load_wav


def merge_donor_clips(tgt_paths: list[Path], output_wav: Path) -> Path:
    """Loads, resamples, concatenates donor audios, and writes to a clean WAV file."""
    tensors = []
    target_sr = 24000  # High sample rate to support both 16k and 24k extractors
    for p in tgt_paths:
        wav, sr = torchaudio.load(str(p))
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        tensors.append(wav)

    combined = torch.cat(tensors, dim=-1)

    # CosyVoice prompt cap (25s max)
    if combined.shape[-1] > target_sr * 25:
        combined = combined[:, :target_sr * 25]

    torchaudio.save(str(output_wav), combined, target_sr)
    return output_wav


def main():
    parser = argparse.ArgumentParser(description="Test CosyVoice 3 Voice Conversion")
    parser.add_argument("--model_dir", type=str, default="pretrained_models/Fun-CosyVoice3-0.5B")
    parser.add_argument("--src", type=Path, required=True, help="Path to source audio clip")
    parser.add_argument("--tgt1", type=Path, required=True, help="Target donor clip 1")
    parser.add_argument("--tgt2", type=Path, default=None, help="Target donor clip 2")
    parser.add_argument("--tgt3", type=Path, default=None, help="Target donor clip 3")
    parser.add_argument("--out", type=Path, default=Path("./test_cosyvoice_out.wav"))
    args = parser.parse_args()

    model_path = Path(args.model_dir)
    if not model_path.is_absolute() and not model_path.exists():
        model_path = REPO_DIR / args.model_dir

    print(f"[INFO] Initializing CosyVoice3 from {model_path}...")
    cosyvoice = CosyVoice3(str(model_path))

    tgt_paths = [args.tgt1]
    if args.tgt2 and args.tgt2.exists():
        tgt_paths.append(args.tgt2)
    if args.tgt3 and args.tgt3.exists():
        tgt_paths.append(args.tgt3)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_prompt:
        tmp_prompt_path = Path(tmp_prompt.name)
        print(f"[INFO] Merging {len(tgt_paths)} donor audio files...")
        merge_donor_clips(tgt_paths, tmp_prompt_path)

        print(f"[INFO] Running Voice Conversion for {args.src.name}...")
        outputs = []
        for output in cosyvoice.inference_vc(str(args.src), str(tmp_prompt_path), stream=False):
            outputs.append(output["tts_speech"])

    if not outputs:
        raise RuntimeError("Inference returned no audio outputs.")

    final_audio = torch.cat(outputs, dim=-1)
    torchaudio.save(str(args.out), final_audio, cosyvoice.sample_rate)
    print(f"[SUCCESS] Synthesized output saved to: {args.out.resolve()}")


if __name__ == "__main__":
    main()