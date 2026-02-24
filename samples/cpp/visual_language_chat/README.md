# MedGemma Medical Image Analysis — C++ Sample

> **⚠️ DISCLAIMER**: MedGemma is designed for RESEARCH AND DEVELOPMENT purposes only.
> Outputs are NOT intended for clinical diagnosis or patient care.
> All results require verification by qualified medical professionals.

## Overview

This C++ sample uses Google's [MedGemma](https://huggingface.co/google/medgemma-1.5-4b-it) vision-language model with the **OpenVINO GenAI VLMPipeline API** to perform interactive medical image analysis (e.g., chest X-rays).

For the full end-to-end setup (clone → download OpenVINO → build → export model → run), see the [top-level README](../../../README.md).

## Files

| File | Description |
|---|---|
| [`medgemma_medical_chat.cpp`](./medgemma_medical_chat.cpp) | Main sample — medical image chat with VLMPipeline |
| [`load_image.cpp`](./load_image.cpp) / [`load_image.hpp`](./load_image.hpp) | Image loading utilities (uses stb_image) |
| [`CMakeLists.txt`](./CMakeLists.txt) | Standalone CMake build (uses `find_package(OpenVINOGenAI)`) |
| [`chest_xray_sample.png`](./chest_xray_sample.png) | Sample test image — Public Domain chest X-ray |
| [`MEDGEMMA_SAMPLE.md`](./MEDGEMMA_SAMPLE.md) | Detailed technical documentation |

## Features

- **VLMPipeline API** — OpenVINO GenAI visual language model inference
- **Predefined Medical Prompts** — `/describe`, `/report`, `/abnormal`, `/anatomy`
- **Multi-turn Conversation** — follow-up questions about the same image
- **Token Streaming** — real-time output via `ov::genai::StreamingStatus` callback
- **Device Selection** — CPU, GPU, or NPU

## Quick Build & Run

> **Prerequisite**: Download OpenVINO GenAI 2026.0.0 first — see [Step 3 in the main README](../../../README.md#step-3-download--extract-openvino-genai-202600).

```powershell
# 1. Set proxy (Intel network only)
$env:http_proxy = "http://proxy-dmz.intel.com:912"
$env:https_proxy = "http://proxy-dmz.intel.com:912"

# 2. Initialize OpenVINO environment (from repo root)
cd C:\working\ai-pc-training
& .\openvino_genai_windows_2026.0.0.0_x86_64\setupvars.ps1

# 3. Build
cd samples\cpp\visual_language_chat
cmake -B build
cmake --build build --config Release
# Output: build\Release\medgemma_medical_chat.exe (~456 KB)

# 4. Run (assumes model already exported — see main README Step 6)
cd C:\working\ai-pc-training
.\samples\cpp\visual_language_chat\build\Release\medgemma_medical_chat.exe `
    medgemma-1.5-4b-it-int4 `
    samples\cpp\visual_language_chat\chest_xray_sample.png `
    GPU
```

## Interactive Commands

| Command | Action |
|---|---|
| `/describe` | General description of the medical image |
| `/report` | Structured radiology findings report |
| `/abnormal` | Focus on detecting abnormalities |
| `/anatomy` | Identify visible anatomical structures |
| `/help` | Show all commands |
| `/quit` or `/exit` | Exit the application |

You can also type any free-form question about the loaded medical image.

## Alternative Build: build_samples_msvc.bat

If using the OpenVINO GenAI package's built-in sample builder:

```powershell
cd openvino_genai_windows_2026.0.0.0_x86_64\samples\cpp
.\build_samples_msvc.bat
# Output: %USERPROFILE%\Documents\Intel\OpenVINO\openvino_cpp_samples_build\intel64\Release\
```

## Test Image

- **File**: `chest_xray_sample.png` (4 MB)
- **Source**: [Wikimedia Commons — Chest X-ray PA](https://commons.wikimedia.org/wiki/File:Chest_Xray_PA_3-8-2010.png)
- **License**: Public Domain

## References

- [MedGemma Model Card](https://huggingface.co/google/medgemma-1.5-4b-it)
- [OpenVINO GenAI Documentation](https://docs.openvino.ai/latest/openvino_genai.html)
- [Optimum Intel Documentation](https://huggingface.co/docs/optimum/intel/index)
- [Detailed Technical Guide](./MEDGEMMA_SAMPLE.md)
