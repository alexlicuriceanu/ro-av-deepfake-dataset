import argparse
from pathlib import Path
from typing import List
import torch
import torchaudio
import yaml
from huggingface_hub import hf_hub_download

from modules.campplus.DTDNN import CAMPPlus
from modules.commons import build_model, load_checkpoint, recursive_munch
from modules.audio import load_audio


def setup_seed_vc(
    config_path: Path,
    ckpt_path: Path,
    campplus_path: Path,
    device: str = "cuda"
):
    print(f"[1/2] Loading Seed-VC configuration and weights on {device}...")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    model_params = recursive_munch(config["model_params"])
    model = build_model(model_params, stage="inference")
    model = load_checkpoint(model, str(ckpt_path), device=device)
    model.to(device)
    model.eval()

    print("[2/2] Loading CAM++ speaker encoder...")
    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_model.load_state_dict(torch.load(str(campplus_path), map_location="cpu"))
    campplus_model.to(device)
    campplus_model.eval()

    return model, campplus_model, config


def prepare_reference_audio(tgt_paths: List[Path], target_sr: int = 22050) -> torch.Tensor:
    """
    Loads and concatenates multiple donor audio clips into a unified reference stream.
    """
    audio_tensors = []
    for path in tgt_paths:
        waveform, sr = torchaudio.load(str(path))
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)
        audio_tensors.append(waveform)

    # Concatenate clips along the time axis
    concat_ref = torch.cat(audio_tensors, dim=-1)
    return concat_ref


def convert_voice(
    src_path: Path,
    tgt_paths: List[Path],
    out_path: Path,
    model,
    campplus_model,
    config,
    diffusion_steps: int = 25,
    cfg_rate: float = 0.7,
    device: str = "cuda",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    target_sr = config["model_params"]["sampling_rate"]

    print(f"Loading source content: {src_path.name}")
    src_audio, src_sr = torchaudio.load(str(src_path))
    if src_audio.shape[0] > 1:
        src_audio = torch.mean(src_audio, dim=0, keepdim=True)
    if src_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=src_sr, new_freq=target_sr)
        src_audio = resampler(src_audio)

    print(f"Aggregating {len(tgt_paths)} donor audio clips for reference timbre...")
    ref_audio = prepare_reference_audio(tgt_paths, target_sr=target_sr)

    src_audio = src_audio.to(device)
    ref_audio = ref_audio.to(device)

    print(f"Running Seed-VC diffusion inference ({diffusion_steps} steps, CFG={cfg_rate})...")
    with torch.no_grad():
        converted_audio = model.inference(
            source=src_audio,
            reference=ref_audio,
            campplus_model=campplus_model,
            n_timesteps=diffusion_steps,
            inference_cfg_rate=cfg_rate,
        )

    # Save converted waveform
    torchaudio.save(str(out_path), converted_audio.cpu(), sample_rate=target_sr)
    print(f"[SUCCESS] Synthesized audio saved to: {out_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Test Seed-VC Single Audio Conversion")
    parser.add_argument("--src", type=Path, required=True, help="Path to source content .wav (to be faked)")
    parser.add_argument("--tgt1", type=Path, required=True, help="Path to primary donor .wav")
    parser.add_argument("--tgt2", type=Path, default=None, help="Optional secondary donor .wav")
    parser.add_argument("--tgt3", type=Path, default=None, help="Optional tertiary donor .wav")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("./test_seed_vc_out.wav"),
        help="Destination path for synthesized .wav",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("checkpoints/config_dit_mel_seed_uvit_whisper_small_wavenet.yml"),
        help="Path to YAML config",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path("checkpoints/DiT_seed_v2_uvit_whisper_small_wavenet.pth"),
        help="Path to Seed-VC model checkpoint",
    )
    parser.add_argument(
        "--campplus",
        type=Path,
        default=Path("checkpoints/campplus_cn_common.bin"),
        help="Path to CAM++ checkpoint",
    )
    parser.add_argument("--steps", type=int, default=25, help="Diffusion sampling steps")
    parser.add_argument("--cfg", type=float, default=0.7, help="Classifier-Free Guidance rate")

    args = parser.parse_args()

    tgt_paths = [args.tgt1]
    if args.tgt2 is not None and args.tgt2.exists():
        tgt_paths.append(args.tgt2)
    if args.tgt3 is not None and args.tgt3.exists():
        tgt_paths.append(args.tgt3)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, campplus_model, config = setup_seed_vc(
        config_path=args.config,
        ckpt_path=args.ckpt,
        campplus_path=args.campplus,
        device=device,
    )

    convert_voice(
        src_path=args.src,
        tgt_paths=tgt_paths,
        out_path=args.out,
        model=model,
        campplus_model=campplus_model,
        config=config,
        diffusion_steps=args.steps,
        cfg_rate=args.cfg,
        device=device,
    )


if __name__ == "__main__":
    main()