import torchaudio
from pathlib import Path
from tqdm import tqdm
import torch
import argparse
import json
from model.ear_vae import EAR_VAE

def main(args):
    indir = args.indir
    model_path = args.model_path
    outdir = args.outdir
    device = args.device
    config_path = args.config

    print(f"Input directory: {indir}")
    print(f"Model path: {model_path}")
    print(f"Output directory: {outdir}")
    print(f"Device: {device}")
    print(f"Config path: {config_path}")
    

    input_path = Path(indir)
    output_path_dir = Path(outdir)
    output_path_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'r') as f:
        vae_gan_model_config = json.load(f)

    print("Loading model...")
    model = EAR_VAE(model_config=vae_gan_model_config).to(device)

    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    print("Model loaded successfully.")

    audios = list(input_path.rglob("*"))
    print(f"Found {len(audios)} audio files to process.")

    with torch.no_grad():
        for audio_path in tqdm(audios, desc="Processing audio files"):
            try:
                gt_y, sr = torchaudio.load(audio_path, backend="ffmpeg")

                if len(gt_y.shape) == 1:
                    gt_y = gt_y.unsqueeze(0)

                # Resample if necessary
                if sr != 44100:
                    resampler = torchaudio.transforms.Resample(sr, 44100).to(device)
                    gt_y = resampler(gt_y)

                gt_y = gt_y.to(device, torch.float32)
                
                # Convert to stereo if mono
                if gt_y.shape[0] == 1:
                    gt_y = torch.cat([gt_y, gt_y], dim=0)

                # Add batch dimension
                gt_y = gt_y.unsqueeze(0)

                fake_audio = model.inference(gt_y)

                output_filename = f"{Path(audio_path).stem}_{Path(model_path).stem}.wav"
                output_path = output_path_dir / output_filename

                fake_audio_processed = fake_audio.squeeze(0).cpu()
                torchaudio.save(output_path, fake_audio_processed, sample_rate=44100, backend="ffmpeg")
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")

                continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run VAE-GAN audio inference.")
    parser.add_argument('--indir', type=str, default='./data', help='Input directory for audio files.')
    parser.add_argument('--model_path', type=str, default='./pretrained_weight/ear_vae_44k.pyt', help='Path to the pretrained model weight.')
    parser.add_argument('--outdir', type=str, default='./results', help='Output directory for generated audio files.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run the model on (e.g., "cuda:0" or "cpu").')
    parser.add_argument('--config', type=str, default='./config/model_config.json', help='Path to the model config file.')
    
    args = parser.parse_args()
    main(args)