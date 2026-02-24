# C++ Visual Language Chat - MedGemma Medical Image Analysis

This sample demonstrates medical image analysis using Google's MedGemma vision-language model with the OpenVINO GenAI VLMPipeline API.

> **⚠️ DISCLAIMER**: MedGemma is designed for RESEARCH AND DEVELOPMENT purposes only. Outputs are NOT intended for clinical diagnosis or patient care. All results require verification by qualified medical professionals.

## Sample File

- [`medgemma_medical_chat.cpp`](./medgemma_medical_chat.cpp) — Medical image analysis using Google's MedGemma model (`google/medgemma-1.5-4b-it`)

## Features

- **VLMPipeline API**: Uses OpenVINO GenAI's VLMPipeline for visual language model inference
- **Medical Disclaimer**: Displays important medical disclaimer at startup
- **Predefined Medical Prompts**:
  - `/describe` — General description of the medical image
  - `/report` — Structured radiology report format
  - `/abnormal` — Focus on abnormality detection
  - `/anatomy` — Identify anatomical structures
- **Multi-turn Conversation**: Supports follow-up questions about the same image
- **Token Streaming**: Real-time output with streaming callback
- **Device Selection**: Supports CPU and GPU inference

## Prerequisites

- **OpenVINO GenAI 2026.0.0** or later
- **Visual Studio 2022** (for Windows build)
- **CMake 3.16+**
- **Hugging Face account** with [MedGemma license accepted](https://huggingface.co/google/medgemma-1.5-4b-it)
- GPU with sufficient memory (recommended) or CPU

## Export MedGemma Model

MedGemma requires accepting the [Health AI Developer Foundation's terms of use](https://huggingface.co/google/medgemma-1.5-4b-it) on Hugging Face before export.

```sh
# Login to Hugging Face (if not already logged in)
huggingface-cli login

# Export MedGemma with INT4 quantization
optimum-cli export openvino -m google/medgemma-1.5-4b-it \
    --task image-text-to-text --weight-format int4 --trust-remote-code \
    medgemma-1.5-4b-it-int4
```

### Exported Model Files

```
medgemma-1.5-4b-it-int4/
├── openvino_language_model.bin          (~2.2 GB)
├── openvino_language_model.xml
├── openvino_vision_embeddings_model.bin (~406 MB)
├── openvino_vision_embeddings_model.xml
├── openvino_text_embeddings_model.bin   (~640 MB)
├── openvino_text_embeddings_model.xml
├── openvino_tokenizer.bin
├── openvino_tokenizer.xml
├── openvino_detokenizer.bin
├── openvino_detokenizer.xml
├── config.json
├── generation_config.json
├── special_tokens_map.json
├── tokenizer_config.json
└── tokenizer.json
```

**Total Size**: ~3.3 GB (INT4 quantized from ~8.6 GB original)

## Build Instructions

### Standalone Build (This Repository)

```powershell
# Set proxy (if behind Intel proxy)
$env:http_proxy = "http://proxy-dmz.intel.com:912"
$env:https_proxy = "http://proxy-dmz.intel.com:912"

# Set up OpenVINO environment
& "C:\path\to\openvino_genai\setupvars.ps1"

# Navigate to sample directory
cd samples\cpp\visual_language_chat

# Build
cmake -B build
cmake --build build --config Release
```

### Using build_samples_msvc.bat (OpenVINO GenAI Package)

If you have the full OpenVINO GenAI package installed, you can also place these files into the corresponding sample directory and build:

```powershell
cd C:\path\to\openvino_genai_windows_2026.0.0.0_x86_64\samples\cpp
.\build_samples_msvc.bat
```

Build output location: `%USERPROFILE%\Documents\Intel\OpenVINO\openvino_cpp_samples_build\`

## Usage

```powershell
# Set up OpenVINO environment
& "C:\path\to\openvino_genai\setupvars.ps1"

# Run with GPU (recommended)
.\medgemma_medical_chat.exe <MODEL_DIR> <IMAGE_PATH> GPU

# Run with CPU
.\medgemma_medical_chat.exe <MODEL_DIR> <IMAGE_PATH> CPU
```

### Example Session

```powershell
.\medgemma_medical_chat.exe "C:\models\medgemma-1.5-4b-it-int4" "chest_xray_sample.png" GPU
```

### Interactive Commands

During the chat session, use these shortcuts:
- `/describe` — Describe the medical image
- `/report` — Generate a structured findings report
- `/abnormal` — Identify abnormalities
- `/anatomy` — Describe visible anatomical structures
- `/help` — Show help
- `/quit` — Exit the application

## Test Image

A sample chest X-ray image is included:

- **File**: `chest_xray_sample.png`
- **Source**: [Wikimedia Commons - Chest X-ray PA](https://commons.wikimedia.org/wiki/File:Chest_Xray_PA_3-8-2010.png)
- **License**: Public Domain

## References

- [MedGemma Model Card](https://huggingface.co/google/medgemma-1.5-4b-it)
- [OpenVINO GenAI Documentation](https://docs.openvino.ai/latest/openvino_genai.html)
- [Optimum Intel Documentation](https://huggingface.co/docs/optimum/intel/index)
- [Detailed Sample Documentation](./MEDGEMMA_SAMPLE.md)
