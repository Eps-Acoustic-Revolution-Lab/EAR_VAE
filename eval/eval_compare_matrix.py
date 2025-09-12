"""
Audio Evaluation Script

This script evaluates the quality of generated audio against ground truth audio
using a variety of metrics, including:
- SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
- Multi-Resolution STFT Loss
- Multi-Resolution Mel-Spectrogram Loss
- Phase Coherence (Per-channel and Inter-channel)
- Loudness metrics (LUFS-I, LRA, True Peak) via ffmpeg.

The script processes a directory of models, where each model directory contains
pairs of reconstructed (_rec.wav) and ground truth (.wav) audio files.
"""

import os
import re
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import auraloss
from tqdm import tqdm

# --- Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAMPLE_RATE = 44100

# --- Metric Definitions ---

# SI-SDR
sisdr_criteria = auraloss.time.SISDRLoss().to(DEVICE)

# Multi-Resolution Mel-Spectrogram Loss
mel_fft_sizes = [4096, 2048, 1024, 512]
mel_win_sizes = mel_fft_sizes
mel_hop_sizes = [i // 4 for i in mel_fft_sizes]
mel_criteria = auraloss.freq.MultiResolutionSTFTLoss(
    fft_sizes=mel_fft_sizes,
    hop_sizes=mel_hop_sizes,
    win_lengths=mel_win_sizes,
    sample_rate=SAMPLE_RATE,
    scale="mel",
    n_bins=64,
    perceptual_weighting=True
).to(DEVICE)

# Multi-Resolution STFT Loss
fft_sizes = [4096, 2048, 1024, 512, 256, 128]
win_sizes = fft_sizes
hop_sizes = [i // 4 for i in fft_sizes]
stft_criteria = auraloss.freq.MultiResolutionSTFTLoss(
    fft_sizes=fft_sizes,
    hop_sizes=hop_sizes,
    win_lengths=win_sizes,
    sample_rate=SAMPLE_RATE,
    perceptual_weighting=True
).to(DEVICE)


def analyze_loudness(file_path: str) -> Optional[Dict[str, float]]:
    """
    Analyzes audio file loudness using ffmpeg's ebur128 filter.

    Args:
        file_path: Path to the audio file.

    Returns:
        A dictionary with LUFS-I, LRA, and True Peak, or None on failure.
    """
    if not Path(file_path).exists():
        logging.warning(f"Loudness analysis skipped: File not found at {file_path}")
        return None

    command = [
        "ffmpeg",
        "-nostats",
        "-i", file_path,
        "-af", "ebur128=peak=true,ametadata=mode=print:file=-",
        "-f", "null",
        "-"
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
        output_text = result.stderr
    except FileNotFoundError:
        logging.error("ffmpeg not found. Please install ffmpeg and ensure it's in your PATH.")
        return None
    except subprocess.CalledProcessError as e:
        logging.error(f"ffmpeg analysis failed for {file_path}. Error: {e.stderr}")
        return None

    loudness_data = {}

    i_match = re.search(r"^\s*I:\s*(-?[\d\.]+)\s*LUFS", output_text, re.MULTILINE)
    if i_match:
        loudness_data['LUFS-I'] = float(i_match.group(1))

    lra_match = re.search(r"^\s*LRA:\s*([\d\.]+)\s*LU", output_text, re.MULTILINE)
    if lra_match:
        loudness_data['LRA'] = float(lra_match.group(1))

    tp_match = re.search(r"Peak:\s*(-?[\d\.]+)\s*dBFS", output_text, re.MULTILINE)
    if tp_match:
        loudness_data['True Peak'] = float(tp_match.group(1))

    if not loudness_data:
        logging.warning(f"Could not parse loudness data for {file_path}.")
        return None

    return loudness_data


class PhaseCoherenceLoss(nn.Module):
    """
    Calculates phase coherence between two audio signals.
    Adapted for stereo and multi-resolution analysis.
    """
    def __init__(self, fft_size, hop_size, win_size, mag_threshold=1e-6, eps=1e-8):
        super().__init__()
        self.fft_size = int(fft_size)
        self.hop_size = int(hop_size)
        self.win_size = int(win_size)
        self.register_buffer("window", torch.hann_window(win_size))
        self.mag_threshold = float(mag_threshold)
        self.eps = float(eps)

    def _to_complex(self, x):
        if torch.is_complex(x):
            return x
        if x.dim() >= 1 and x.size(-1) == 2:
            return torch.complex(x[..., 0], x[..., 1])
        raise ValueError("Input must be complex or real/imag tensor.")

    def _stereo_stft(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        B, C, T = x.shape
        stft = torch.stft(x.reshape(B * C, T),
                          n_fft=self.fft_size,
                          hop_length=self.hop_size,
                          win_length=self.win_size,
                          window=self.window,
                          return_complex=True)
        return stft.view(B, C, -1, stft.size(-1))

    def forward(self, pred, target):
        pred_stft = self._stereo_stft(pred)
        target_stft = self._stereo_stft(target)

        pred_stft = self._to_complex(pred_stft)
        target_stft = self._to_complex(target_stft)

        B, C, F, T = pred_stft.shape
        
        # magnitudes and weights
        mag_pred = torch.abs(pred_stft)
        mag_target = torch.abs(target_stft)
        weights = mag_pred * mag_target
        mask = (weights > self.mag_threshold).to(weights.dtype)
        weights_masked = weights * mask

        # phase difference Δφ = angle(pred) - angle(target)
        delta = torch.angle(pred_stft) - torch.angle(target_stft)
        # phasor e^{jΔφ}
        phasor = torch.complex(torch.cos(delta), torch.sin(delta))

        # weighted vector sum across frequency axis
        num = torch.sum(weights_masked * phasor, dim=2) # [B, C, T], complex
        den = torch.sum(weights_masked, dim=2).clamp_min(self.eps)
        coherence_per_bin = torch.abs(num) / den

        # pool across time (energy-weighted mean) -> per-channel scalar
        # weight time pooling by per-frame energy sum to emphasize active frames
        frame_energy = torch.sum(weights_masked, dim=2)
        frame_energy_sum = torch.sum(frame_energy, dim=2).clamp_min(self.eps)
        
        # energy-weighted average over time
        coherence_chan = torch.sum(coherence_per_bin * frame_energy, dim=2) / frame_energy_sum
        
        # mean across batch
        per_channel_coherence = coherence_chan.mean(dim=0)

        inter_coherence = None
        if C >= 2:
            Lp, Rp = pred_stft[:, 0], pred_stft[:, 1]
            Lt, Rt = target_stft[:, 0], target_stft[:, 1]
            
            # inter-channel phase: angle(L) - angle(R)  <=> angle(L * conj(R))
            inter_delta = torch.angle(Lp * torch.conj(Rp)) - torch.angle(Lt * torch.conj(Rt))
            inter_weights = torch.abs(Lp) * torch.abs(Rp)
            inter_mask = (inter_weights > self.mag_threshold).to(inter_weights.dtype)
            inter_weights_masked = inter_weights * inter_mask
            inter_phasor = torch.complex(torch.cos(inter_delta), torch.sin(inter_delta))
            inter_num = torch.sum(inter_weights_masked * inter_phasor, dim=1)
            inter_den = torch.sum(inter_weights_masked, dim=1).clamp_min(self.eps)
            inter_coh_time = torch.abs(inter_num) / inter_den

            # pool across time weighted by energy
            inter_frame_energy = torch.sum(inter_weights_masked, dim=1)
            inter_energy_sum = inter_frame_energy.sum(dim=1).clamp_min(self.eps)
            inter_coh_b = (inter_coh_time * inter_frame_energy).sum(dim=1) / inter_energy_sum
            inter_coherence = inter_coh_b.mean()

        return {
            "per_channel_coherence": per_channel_coherence.detach().cpu(),
            "interchannel_coherence": (inter_coherence.detach().cpu() if inter_coherence is not None else None),
        }


class MultiResolutionPhaseCoherenceLoss(nn.Module):
    def __init__(self, fft_sizes, hop_sizes, win_sizes):
        super().__init__()
        self.criteria = nn.ModuleList([
            PhaseCoherenceLoss(fft, hop, win) for fft, hop, win in zip(fft_sizes, hop_sizes, win_sizes)
        ])

    def forward(self, pred, target):
        results = [criterion(pred, target) for criterion in self.criteria]
        
        per_channel = torch.stack([r["per_channel_coherence"] for r in results]).mean(dim=0)
        inter_items = [r["interchannel_coherence"] for r in results if r["interchannel_coherence"] is not None]
        inter_channel = torch.stack(inter_items).mean() if inter_items else None

        return {"per_channel_coherence": per_channel, "interchannel_coherence": inter_channel}

phase_coherence_criteria = MultiResolutionPhaseCoherenceLoss(
    fft_sizes=mel_fft_sizes, hop_sizes=mel_hop_sizes, win_sizes=mel_win_sizes
).to(DEVICE)


def find_audio_pairs(model_path: Path) -> List[Tuple[Path, Path]]:
    """Finds pairs of reconstructed and ground truth audio files."""
    rec_files = sorted(model_path.glob("*_vae_rec.wav"))
    pairs = []
    for rec_file in rec_files:
        gt_file = model_path / rec_file.name.replace("_vae_rec.wav", ".wav")
        if gt_file.exists():
            pairs.append((rec_file, gt_file))
        else:
            logging.warning(f"Ground truth file not found for {rec_file.name}")
    return pairs


def evaluate_pair(rec_path: Path, gt_path: Path) -> Optional[Dict[str, float]]:
    """Evaluates a single pair of audio files."""
    try:
        gen_wav, gen_sr = torchaudio.load(rec_path, backend="ffmpeg")
        gt_wav, gt_sr = torchaudio.load(gt_path, backend="ffmpeg")

        if gen_sr != SAMPLE_RATE:
            gen_wav = torchaudio.transforms.Resample(gen_sr, SAMPLE_RATE)(gen_wav)
        if gt_sr != SAMPLE_RATE:
            gt_wav = torchaudio.transforms.Resample(gt_sr, SAMPLE_RATE)(gt_wav)

        # Trim to same length
        if gen_wav.shape[-1] != gt_wav.shape[-1]:
            logging.info(f"Shape Mismatched, Trimming audio files to the same length: {rec_path.name}, {gt_path.name}")
            min_len = min(gen_wav.shape[-1], gt_wav.shape[-1])
            gen_wav, gt_wav = gen_wav[:, :min_len], gt_wav[:, :min_len]

        gen_wav, gt_wav = gen_wav.to(DEVICE).unsqueeze(0), gt_wav.to(DEVICE).unsqueeze(0)

        metrics = {}
        metrics['sisdr'] = -sisdr_criteria(gen_wav, gt_wav).item()
        metrics['mel_distance'] = mel_criteria(gen_wav, gt_wav).item()
        metrics['stft_distance'] = stft_criteria(gen_wav, gt_wav).item()

        phase_metrics = phase_coherence_criteria(gen_wav, gt_wav)
        metrics['per_channel_coherence'] = phase_metrics["per_channel_coherence"].mean().item()
        if phase_metrics["interchannel_coherence"] is not None:
            metrics['interchannel_coherence'] = phase_metrics["interchannel_coherence"].item()

        return metrics
    except Exception as e:
        logging.error(f"Error processing pair {rec_path.name}, {gt_path.name}: {e}")
        return None


def process_model(model_path: Path, force_eval: bool = False, echo=True):
    """Processes all audio pairs for a given model."""
    logging.info(f"Processing model: {model_path.name}")
    results_file = model_path / "evaluation_results.json"
    
    if results_file.exists() and not force_eval:
        logging.info(f"Results already exist for {model_path.name}, skipping.")
        return

    audio_pairs = find_audio_pairs(model_path)
    if not audio_pairs:
        logging.warning(f"No valid audio pairs found for {model_path.name}.")
        return

    all_metrics = []
    gen_loudness_data, gt_loudness_data = [], []

    with torch.no_grad():
        for rec_path, gt_path in tqdm(audio_pairs, desc=f"Evaluating {model_path.name}"):
            pair_metrics = evaluate_pair(rec_path, gt_path)
            if pair_metrics:
                all_metrics.append(pair_metrics)
            
            gen_loudness = analyze_loudness(str(rec_path))
            if gen_loudness:
                gen_loudness_data.append(gen_loudness)
            
            gt_loudness = analyze_loudness(str(gt_path))
            if gt_loudness:
                gt_loudness_data.append(gt_loudness)
            
            if echo:
                logging.info(f"Metrics for {rec_path.name}: {pair_metrics}")
                if gen_loudness:
                    logging.info(f"Generated Loudness: {gen_loudness}")
                if gt_loudness:
                    logging.info(f"Ground Truth Loudness: {gt_loudness}")

    if not all_metrics:
        logging.warning(f"No metrics could be calculated for {model_path.name}.")
        return

    # Aggregate results
    summary = {"model_name": model_path.name, "file_count": len(all_metrics)}
    
    # Average objective metrics
    metric_keys = all_metrics[0].keys()
    for key in metric_keys:
        valid_values = [m[key] for m in all_metrics if key in m]
        if valid_values:
            summary[f"avg_{key}"] = float(np.mean(valid_values))

    # Average loudness metrics
    def _avg_loudness(data: List[Dict[str, float]], prefix: str):
        if not data: return
        for key in data[0].keys():
            values = [d[key] for d in data if key in d]
            if values:
                summary[f"avg_{prefix}_{key.lower().replace(' ', '_')}"] = float(np.mean(values))

    _avg_loudness(gen_loudness_data, "gen")
    _avg_loudness(gt_loudness_data, "gt")

    # Save results
    logging.info(f"Saving results for {model_path.name} to {results_file}")
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Also save a human-readable version
    with open(model_path / "evaluation_summary.txt", "w") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")


def main():
    parser = argparse.ArgumentParser(description="Run evaluation on generated audio.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Root directory containing model output folders."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-evaluation even if results files exist."
    )

    parser.add_argument(
        "--echo",
        action="store_true",
        help="Echo per-file metrics to console during evaluation."
    )
    args = parser.parse_args()

    root_path = Path(args.input_dir)
    if not root_path.is_dir():
        logging.error(f"Input directory not found: {root_path}")
        sys.exit(1)

    model_paths = [p for p in root_path.iterdir() if p.is_dir() and not p.name.startswith('.')]
    
    logging.info(f"Found {len(model_paths)} model(s) to evaluate: {[p.name for p in model_paths]}")

    for model_path in sorted(model_paths):
        process_model(model_path, args.force, args.echo)

    logging.info("Evaluation complete.")


if __name__ == "__main__":
    main()