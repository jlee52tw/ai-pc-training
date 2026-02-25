# MedASR Medical Speech Recognition Sample

A C++ sample demonstrating medical speech-to-text transcription using Google's
[MedASR](https://huggingface.co/google/medasr) model with OpenVINO Runtime.

## Overview

MedASR is a speech-to-text model based on the [Conformer architecture](https://arxiv.org/abs/2005.08100)
pre-trained for medical dictation. It uses CTC (Connectionist Temporal Classification) decoding
and is optimized for medical terminology including radiology, internal medicine, and family medicine.

**Key specifications:**
- Architecture: Conformer (CTC)
- Parameters: 105M
- Input: 16kHz mono audio
- Output: Text transcription
- Specialization: Medical dictation (radiology, clinical notes)

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Windows 10/11 | x86_64 | Tested on Windows 11 |
| Visual Studio 2022 | with C++ Desktop workload | MSVC compiler |
| CMake | 3.16+ | Included with VS 2022 |
| Python | 3.10–3.12 | For model export only |
| [Hugging Face account](https://huggingface.co/) | — | Must accept [MedASR license](https://huggingface.co/google/medasr) |
| GPU (Intel/discrete) | — | Recommended; CPU also works |

## Quick Start

### Step 1: Set Up OpenVINO Environment

From the repository root:

```powershell
# Download and set up OpenVINO GenAI (if not already done)
.\setup.ps1

# Or manually set up environment
& .\openvino_genai_windows_2026.0.0.0_x86_64\setupvars.ps1
```

### Step 2: Set Proxy (Intel Network Only)

```powershell
$env:http_proxy = "http://proxy-dmz.intel.com:912"
$env:https_proxy = "http://proxy-dmz.intel.com:912"
$env:no_proxy = ".intel.com,intel.com,localhost,127.0.0.1"
```

### Step 3: Export MedASR Model

Create a Python virtual environment and export the model:

```powershell
# From repo root
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install --proxy $env:http_proxy torch openvino transformers safetensors huggingface_hub

# Install transformers from git (MedASR requires transformers 5.0.0+)
pip install --proxy $env:http_proxy git+https://github.com/huggingface/transformers.git@65dc261512cbdb1ee72b88ae5b222f2605aad8e5

# Login to Hugging Face (needed for gated model access)
huggingface-cli login

# Export model to OpenVINO IR format (from the asr sample directory)
cd samples\cpp\asr
python export_model.py --output medasr-openvino
```

The export creates a `medasr-openvino/` directory containing:
- `medasr_model.xml` / `.bin` — OpenVINO IR model
- `mel_filters.bin` — Mel filterbank matrix for feature extraction
- `vocab_id2token.json` — Vocabulary for CTC decoding

> **Note**: You must first accept the [MedASR license](https://huggingface.co/google/medasr)
> on Hugging Face, or the download will fail.

### Step 4: Build the Sample

```powershell
cd samples\cpp\asr
cmake -B build
cmake --build build --config Release
```

Build output: `build\Release\medasr_medical_asr.exe`

### Step 5: Download Test Audio

MedASR provides a test audio file in the Hugging Face repository:

```powershell
# Download test audio from HuggingFace (from the asr sample directory)
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('google/medasr', 'test_audio.wav', local_dir='.')"
```

Or use any 16kHz mono WAV file with medical dictation content.

### Step 6: Run the Demo

```powershell
# Make sure OpenVINO environment is set (from repo root)
& .\openvino_genai_windows_2026.0.0.0_x86_64\setupvars.ps1

# Run from the asr sample directory
cd samples\cpp\asr

# Run with CPU
.\build\Release\medasr_medical_asr.exe medasr-openvino test_audio.wav CPU

# Run with GPU (faster)
.\build\Release\medasr_medical_asr.exe medasr-openvino test_audio.wav GPU
```

### Expected Output

```
================================================================
|              IMPORTANT DISCLAIMER                            |
================================================================
| MedASR is designed for RESEARCH AND DEVELOPMENT only.        |
| Outputs are NOT intended for clinical diagnosis or treatment.|
| All results require verification by qualified professionals. |
================================================================

Loading vocabulary...
Loaded vocabulary: 512 tokens
Loading mel filterbank...
Reading audio: test_audio.wav
Audio duration: 43.8 seconds (700800 samples)
Computing mel spectrogram...
Mel spectrogram: 4378 frames x 128 bins
Device: CPU
Initializing OpenVINO Runtime...
Loading model: medasr_model.xml
Running inference...
Output shape: [1, 1092, 512]

================================================================
Transcription:
================================================================
[EXAM TYPE] CT chest PE protocol {period} [INDICATION] 54-year-old female,
shortness of breath, evaluate for PE {period} [TECHNIQUE] Standard protocol
{period} [FINDINGS] {colon} Pulmonary vasculature {colon} The main PA is patent
{period} There are filling defects in the segmental branches of the right lower
lobe {comma} compatible with acute PE {period} No saddle embolus {period} ...
================================================================
Inference time: 728 ms
Audio duration: 43.8 s
Real-time factor: 0.02x
```

## Architecture

```
Audio (16kHz WAV)
    │
    ▼
[WAV Reader] ──── dr_wav library
    │
    ▼
[Mel Spectrogram] ── STFT → power spectrum → mel filterbank → log
    │                  (n_fft=512, hop=160, win=400, 128 mel bins)
    ▼
[OpenVINO Runtime] ── Conformer-CTC model inference
    │
    ▼
[CTC Decoder] ──── argmax → collapse repeats → remove blanks
    │               (SentencePiece tokenizer, 512 vocab)
    ▼
Transcription text
```

## Files

| File | Description |
|---|---|
| `medasr_medical_asr.cpp` | Main application: model loading, inference, CTC decoding |
| `audio_utils.hpp/cpp` | WAV reader + mel spectrogram computation (FFT, mel filterbank) |
| `export_model.py` | Python script to export MedASR to OpenVINO IR |
| `CMakeLists.txt` | Build configuration |
| `README.md` | This file |

## Performance Benchmarks (from Google)

| Dataset | MedASR (greedy) | MedASR + LM | Whisper v3 Large |
|---|---|---|---|
| RAD-DICT (radiology) | 6.6% WER | **4.6% WER** | 25.3% WER |
| GENERAL-DICT (general medicine) | 9.3% WER | **6.9% WER** | 33.1% WER |
| FM-DICT (family medicine) | 8.1% WER | **5.8% WER** | 32.5% WER |

## Limitations

- **English only**: All training data is in English
- **Microphone quality**: Best performance with high-quality microphones
- **Speaker diversity**: Primarily trained on US English speakers
- **Medical terminology**: May not cover all medications or procedures

## References

- Model: [google/medasr](https://huggingface.co/google/medasr)
- GitHub: [google-health/medasr](https://github.com/google-health/medasr)
- Paper: [LAST: Scalable Lattice-Based Speech Modelling in JAX](https://arxiv.org/pdf/2304.13134)
- Architecture: [Conformer](https://arxiv.org/abs/2005.08100)

## Disclaimer

MedASR is intended for **research and development purposes only**. It should NOT
be used for clinical diagnosis or patient care decisions. All outputs require
verification by qualified medical professionals.
