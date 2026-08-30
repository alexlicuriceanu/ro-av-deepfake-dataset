import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import librosa
import torchaudio
import yaml
from pathlib import Path
from typing import Dict
from tqdm import tqdm

os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'
import warnings
warnings.simplefilter('ignore')

SEED_VC_ROOT = Path(__file__).resolve().parent / "seed-vc"
sys.path.insert(0, str(SEED_VC_ROOT))

from modules.commons import str2bool
from hf_utils import load_custom_model_from_hf

# --- OFFICIAL HELPER FUNCTIONS ---
def adjust_f0_semitones(f0_sequence, n_semitones):
    factor = 2 ** (n_semitones / 12)
    return f0_sequence * factor

def crossfade(chunk1, chunk2, overlap):
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    if len(chunk2) < overlap:
        chunk2[:overlap] = chunk2[:overlap] * fade_in[:len(chunk2)] + (chunk1[-overlap:] * fade_out)[:len(chunk2)]
    else:
        chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
    return chunk2

# Wrap the official model loader to accept a dummy args object
def load_models_official(device, fp16):
    class DummyArgs:
        def __init__(self):
            self.fp16 = fp16
            self.f0_condition = False
            self.checkpoint = None
            self.config = None

    args = DummyArgs()
    
    # IMPORT the user's load_models from the official script logic
    # To keep it self-contained, we recreate the exact initialization here:
    
    dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(
        "Plachta/Seed-VC",
        "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
        "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
    )
    f0_fn = None
    
    config = yaml.safe_load(open(dit_config_path, "r"))
    from modules.commons import recursive_munch, build_model, load_checkpoint
    
    model_params = recursive_munch(config["model_params"])
    model_params.dit_type = 'DiT'
    model = build_model(model_params, stage="DiT")
    
    model, _, _, _ = load_checkpoint(
        model, None, dit_checkpoint_path, load_only_params=True, ignore_modules=[], is_distributed=False
    )
    for key in model:
        model[key].eval().to(device)
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    from modules.campplus.DTDNN import CAMPPlus
    campplus_ckpt_path = load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin", config_filename=None)
    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
    campplus_model.eval().to(device)

    # BigVGAN Vocoder (Default for this config)
    from modules.bigvgan import bigvgan
    bigvgan_name = model_params.vocoder.name
    bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
    bigvgan_model.remove_weight_norm()
    vocoder_fn = bigvgan_model.eval().to(device)

    # Whisper Semantic Extractor
    from transformers import AutoFeatureExtractor, WhisperModel
    whisper_name = model_params.speech_tokenizer.name
    whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(device)
    del whisper_model.decoder
    whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

    def semantic_fn(waves_16k):
        ori_inputs = whisper_feature_extractor([waves_16k.squeeze(0).cpu().numpy()], return_tensors="pt", return_attention_mask=True)
        ori_input_features = whisper_model._mask_input_features(ori_inputs.input_features, attention_mask=ori_inputs.attention_mask).to(device)
        with torch.no_grad():
            ori_outputs = whisper_model.encoder(
                ori_input_features.to(whisper_model.encoder.dtype), head_mask=None, output_attentions=False, output_hidden_states=False, return_dict=True,
            )
        S_ori = ori_outputs.last_hidden_state.to(torch.float32)
        S_ori = S_ori[:, :waves_16k.size(-1) // 320 + 1]
        return S_ori

    sr = config["preprocess_params"]["sr"]
    mel_fn_args = {
        "n_fft": config['preprocess_params']['spect_params']['n_fft'],
        "win_size": config['preprocess_params']['spect_params']['win_length'],
        "hop_size": config['preprocess_params']['spect_params']['hop_length'],
        "num_mels": config['preprocess_params']['spect_params']['n_mels'],
        "sampling_rate": sr,
        "fmin": config['preprocess_params']['spect_params'].get('fmin', 0),
        "fmax": None if config['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
        "center": False
    }
    from modules.audio import mel_spectrogram
    to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

    return model, semantic_fn, f0_fn, vocoder_fn, campplus_model, to_mel, mel_fn_args


def save_manifest_safe(manifest: Dict, manifest_path: Path):
    temp_path = manifest_path.with_suffix(".json.tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    temp_path.replace(manifest_path)


def main():
    parser = argparse.ArgumentParser(description="Seed-VC True Batch Inference")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to master manifest")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--diffusion-steps", type=int, default=25)
    parser.add_argument("--length-adjust", type=float, default=1.0)
    parser.add_argument("--inference-cfg-rate", type=float, default=0.7)
    parser.add_argument("--fp16", type=str2bool, default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading official models to {device}...")
    model, semantic_fn, f0_fn, vocoder_fn, campplus_model, mel_fn, mel_fn_args = load_models_official(device, args.fp16)
    sr = mel_fn_args['sampling_rate']

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    seed_tasks = {k: v for k, v in manifest.items() if v.get("audio_manipulation", {}).get("assigned_method") == "seed_vc"}
    
    if not seed_tasks:
        print("[INFO] No Seed-VC tasks found in the manifest.")
        return

    skipped_count = 0
    donor_cache = {}
    pbar = tqdm(seed_tasks.items(), desc="Generating Seed-VC")

    for key, entry in pbar:
        manip_data = entry["audio_manipulation"]
        
        # Resume safety
        if "faked_audio_path" in manip_data:
            existing_out = Path(manip_data["faked_audio_path"])
            if existing_out.exists() and existing_out.stat().st_size > 0:
                continue

        source_path = entry["audio_path"]
        tgt_paths = manip_data["donor_audio_paths"]
        donor_vid = manip_data.get("donor_video_id")
        out_path = args.out_dir / entry["video_id"] / Path(source_path).name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        pbar.set_postfix({"current": Path(source_path).name, "skipped": skipped_count})

        try:
            # 1. Load source audio
            source_audio = librosa.load(source_path, sr=sr)[0]

            # 2. Load or compute concatenated target donor
            if donor_vid in donor_cache:
                ref_audio = donor_cache[donor_vid]
            else:
                # IN-MEMORY MERGE: Equivalent to ffmpeg concat, but instantly in RAM
                ref_audio_list = [librosa.load(p, sr=sr)[0] for p in tgt_paths]
                ref_audio = np.concatenate(ref_audio_list)
                donor_cache[donor_vid] = ref_audio

            hop_length = 256
            max_context_window = sr // hop_length * 30
            overlap_frame_len = 16
            overlap_wave_len = overlap_frame_len * hop_length

            source_tensor = torch.tensor(source_audio).unsqueeze(0).float().to(device)
            ref_tensor = torch.tensor(ref_audio[:sr * 25]).unsqueeze(0).float().to(device)

            # ----- EXACT OFFICIAL INFERENCE PIPELINE -----
            with torch.no_grad():
                converted_waves_16k = torchaudio.functional.resample(source_tensor, sr, 16000)
                
                if converted_waves_16k.size(-1) <= 16000 * 30:
                    S_alt = semantic_fn(converted_waves_16k)
                else:
                    overlapping_time = 5 
                    S_alt_list, buffer = [], None
                    traversed_time = 0
                    while traversed_time < converted_waves_16k.size(-1):
                        if buffer is None:
                            chunk = converted_waves_16k[:, traversed_time:traversed_time + 16000 * 30]
                        else:
                            chunk = torch.cat([buffer, converted_waves_16k[:, traversed_time:traversed_time + 16000 * (30 - overlapping_time)]], dim=-1)
                        S_alt_chunk = semantic_fn(chunk)
                        if traversed_time == 0:
                            S_alt_list.append(S_alt_chunk)
                        else:
                            S_alt_list.append(S_alt_chunk[:, 50 * overlapping_time:])
                        buffer = chunk[:, -16000 * overlapping_time:]
                        traversed_time += 30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
                    S_alt = torch.cat(S_alt_list, dim=1)

                ori_waves_16k = torchaudio.functional.resample(ref_tensor, sr, 16000)
                S_ori = semantic_fn(ori_waves_16k)
                mel = mel_fn(source_tensor.float())
                mel2 = mel_fn(ref_tensor.float())

                target_lengths = torch.LongTensor([int(mel.size(2) * args.length_adjust)]).to(mel.device)
                target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

                feat2 = torchaudio.compliance.kaldi.fbank(ori_waves_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
                feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
                style2 = campplus_model(feat2.unsqueeze(0))

                cond, _, _, _, _ = model.length_regulator(S_alt, ylens=target_lengths, n_quantizers=3, f0=None)
                prompt_condition, _, _, _, _ = model.length_regulator(S_ori, ylens=target2_lengths, n_quantizers=3, f0=None)

                max_source_window = max_context_window - mel2.size(2)
                processed_frames = 0
                generated_wave_chunks = []

                while processed_frames < cond.size(1):
                    chunk_cond = cond[:, processed_frames:processed_frames + max_source_window]
                    is_last_chunk = processed_frames + max_source_window >= cond.size(1)
                    cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
                    
                    with torch.autocast(device_type=device.type, dtype=torch.float16 if args.fp16 else torch.float32):
                        vc_target = model.cfm.inference(
                            cat_condition, torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                            mel2, style2, None, args.diffusion_steps, inference_cfg_rate=args.inference_cfg_rate
                        )
                        vc_target = vc_target[:, :, mel2.size(-1):]
                        
                    vc_wave = vocoder_fn(vc_target.float()).squeeze()[None, :]
                    
                    if processed_frames == 0:
                        if is_last_chunk:
                            generated_wave_chunks.append(vc_wave[0].cpu().numpy())
                            break
                        generated_wave_chunks.append(vc_wave[0, :-overlap_wave_len].cpu().numpy())
                        previous_chunk = vc_wave[0, -overlap_wave_len:]
                        processed_frames += vc_target.size(2) - overlap_frame_len
                    elif is_last_chunk:
                        output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len)
                        generated_wave_chunks.append(output_wave)
                        processed_frames += vc_target.size(2) - overlap_frame_len
                        break
                    else:
                        output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0, :-overlap_wave_len].cpu().numpy(), overlap_wave_len)
                        generated_wave_chunks.append(output_wave)
                        previous_chunk = vc_wave[0, -overlap_wave_len:]
                        processed_frames += vc_target.size(2) - overlap_frame_len

                vc_wave = torch.tensor(np.concatenate(generated_wave_chunks))[None, :].float()

            torchaudio.save(str(out_path), vc_wave.cpu(), sr)

            manifest[key]["audio_manipulation"]["faked_audio_path"] = str(out_path.resolve())
            save_manifest_safe(manifest, args.manifest)

        except Exception as e:
            skipped_count += 1
            tqdm.write(f"[WARN] Skipped {Path(source_path).name}: {str(e)}")
            continue

    print(f"[SUCCESS] Batch processing finished. Completed: {len(seed_tasks) - skipped_count}, Skipped: {skipped_count}")

if __name__ == "__main__":
    main()