# εar-VAE: High Fidelity Music Reconstruction Model

This repository contains the official inference code for εar-VAE, aa 44.1 kHz music signal reconstruction model that rethinks and optimizes VAE training for audio. It targets two common weaknesses in existing open-source VAEs—phase accuracy and stereophonic spatial representation—by aligning objectives with auditory perception and introducing phase-aware training. Experiments show substantial improvements across diverse metrics, with particular strength in high-frequency harmonics and spatial characteristics.

<p align="center">
<img src="./images/all_compares.jpg" width=90%>
<img src="./images/table.png" width=90%>
</p>

<p align="center">
<em>Upper: Ablation study across our training components.</em> <em>Down: Cross-model metric comparison on the evaluation dataset.</em>
</p>

Why εar-VAE:
- 🎧 Perceptual alignment: A K-weighting perceptual filter is applied before loss computation to better match human hearing.
- 🔁 Phase-aware objectives: Two novel phase losses
  - Stereo Correlation Loss for robust inter-channel coherence.
  - Phase-Derivative Loss using Instantaneous Frequency and Group Delay for phase precision.
- 🌈 Spectral supervision paradigm: Magnitude supervised across MSLR (Mid/Side/Left/Right) components, while phase is supervised only by LR (Left/Right), improving stability and fidelity.
- 📈 44.1 kHz performance: Outperforms leading open-source models across diverse metrics.

## 1. Installation

Follow these steps to set up the environment and install the necessary dependencies.

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd ear_vae
    ```

2.  **Create and activate a conda environment:**
    ```bash
    conda create -n ear_vae python=3.8
    conda activate ear_vae
    ```

3.  **Run the installation script:**
    
    This script will install the remaining dependencies.
    ```bash
    bash install_requirements.sh
    ```
    This will install:
    - `descript-audio-codec`
    - `alias-free-torch`
    - `ffmpeg < 7` (via conda)
    
4.  **Download the model weight:**

    You could download the model checkpoint from **[Hugging Face](https://huggingface.co/earlab/EAR_VAE)**
## 2. Usage

The `inference.py` script is used to process audio files from an input directory and save the reconstructed audio to an output directory.

### Running Inference

You can run the inference with the following command:

```bash
python inference.py --indir <input_directory> --outdir <output_directory> --model_path <path_to_model> --device <device>
```

### Command-Line Arguments

-   `--indir`: (Optional) Path to the input directory containing audio files. Default: `./data`.
-   `--outdir`: (Optional) Path to the output directory where reconstructed audio will be saved. Default: `./results`.
-   `--model_path`: (Optional) Path to the pretrained model weights (`.pyt` file). Default: `./pretrained_weight/ear_vae_44k.pyt`.
-   `--device`: (Optional) The device to run the model on (e.g., `cuda:0` or `cpu`). Defaults to `cuda:0` if available, otherwise `cpu`.

### Example

1.  Place your input audio files (e.g., `.wav`, `.mp3`) into the `data/` directory.
2.  Run the inference script:

    ```bash
    python inference.py
    ```
    This will use the default paths. The reconstructed audio files will be saved in the `results/` directory.

## 3. Project Structure

```
.
├── README.md               # This file
├── config/                 # For model configurations
│   └── model_config.json
├── data/                   # Default directory for input audio files
├── eval/                   # Scripts for model evaluation
│   ├── eval_compare_matrix.py
│   ├── install_requirements.sh
│   └── README.md
├── inference.py            # Main script for running audio reconstruction
├── install_requirements.sh # Installation script for dependencies
├── model/                  # Contains the model architecture code
│   ├── sa2vae.py
│   ├── transformer.py
│   └── vaegan.py
├── pretrained_weight/      # Directory for pretrained model weights
│   └── your_weight_here
```

## 4. Model Details

The model is a Variational Autoencoder with a Generative Adversarial Network (VAE-GAN) structure.
-   **Encoder**: An Oobleck-style encoder that downsamples the input audio into a latent representation.
-   **Bottleneck**: A VAE bottleneck that introduces a probabilistic latent space, sampling from a learned mean and variance.
-   **Decoder**: An Oobleck-style decoder that upsamples the latent representation back into an audio waveform.
-   **Transformer**: A Continuous Transformer can optionally be placed in the bottleneck to further process the latent sequence.

This architecture allows for efficient and high-quality audio reconstruction.

## 5. Evaluation

The `eval/` directory contains scripts to evaluate the model's reconstruction performance using objective metrics.

### Evaluation Prerequisites

1.  **Install Dependencies**: The evaluation script has its own set of dependencies. Install them by running the script in the `eval` directory:
    ```bash
    bash eval/install_requirements.sh
    ```
    This will install libraries such as `auraloss`.

2.  **FFmpeg**: The script uses `ffmpeg` for loudness analysis. Make sure `ffmpeg` is installed and available in your system's PATH. You can install it via conda:
    ```bash
    conda install -c conda-forge 'ffmpeg<7'
    ```

### Running Evaluation

The `eval_compare_matrix.py` script compares the reconstructed audio with the original ground truth files and computes various metrics.

For more details on the evaluation metrics and options, refer to the `eval/README.md` file.

## 6. Acknowledgements

This project builds upon the work of several open-source projects. We would like to extend our special thanks to:

-   **[Stability AI's Stable Audio Tools](https://github.com/Stability-AI/stable-audio-tools)**: For providing a foundational framework and tools for audio generation.
-   **[Descript's Audio Codec](https://github.com/descriptinc/descript-audio-codec)**: For the weight-normed convolusional layers

Their contributions have been invaluable to the development of εar-VAE.


