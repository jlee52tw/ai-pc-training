# AI PC Training — April 2026

Training materials and samples for AI on PC development using Intel OpenVINO GenAI.

## Contents

### MedGemma Medical Image Analysis (OpenVINO GenAI C++)

A C++ sample demonstrating medical image analysis using Google's [MedGemma](https://huggingface.co/google/medgemma-1.5-4b-it) vision-language model with the OpenVINO GenAI VLMPipeline API.

- **Sample code**: [`samples/cpp/visual_language_chat/`](samples/cpp/visual_language_chat/)
- **Quick start**: [`samples/cpp/visual_language_chat/README.md`](samples/cpp/visual_language_chat/README.md)
- **Detailed guide**: [`samples/cpp/visual_language_chat/MEDGEMMA_SAMPLE.md`](samples/cpp/visual_language_chat/MEDGEMMA_SAMPLE.md)

Features:
- Interactive medical image Q&A with predefined prompts (`/describe`, `/report`, `/abnormal`, `/anatomy`)
- Multi-turn conversation with streaming token output
- GPU and CPU inference support
- Includes sample chest X-ray test image (Public Domain)

---

## End-to-End Setup Guide (Windows)

Follow these steps from a fresh clone to a working demo. All commands are PowerShell.

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Windows 10/11 | x86_64 | Tested on Windows 11 |
| Visual Studio 2022 | with C++ Desktop workload | MSVC compiler |
| CMake | 3.16+ | Included with VS 2022 |
| Python | 3.10–3.12 | For model export only |
| [Hugging Face account](https://huggingface.co/) | — | Must accept [MedGemma license](https://huggingface.co/google/medgemma-1.5-4b-it) |
| GPU (Intel/discrete) | — | Recommended; CPU also works |

### Step 1: Clone the Repository

```powershell
git clone https://github.com/jlee52tw/ai-pc-training.git
cd ai-pc-training
git checkout openvino-medgemma
```

### Step 2: Set Proxy (Intel Network Only)

Skip this step if you're not behind the Intel corporate proxy.

```powershell
$env:http_proxy = "http://proxy-dmz.intel.com:912"
$env:https_proxy = "http://proxy-dmz.intel.com:912"
$env:no_proxy = ".intel.com,intel.com,localhost,127.0.0.1"
```

### Step 3: Download & Extract OpenVINO GenAI 2026.0.0

Download the pre-built OpenVINO GenAI package (~223 MB):

```powershell
# Option A: Use the setup script (handles download, extract, and env setup)
.\setup.ps1

# Option B: Manual download
Invoke-WebRequest -Uri "https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2026.0/windows/openvino_genai_windows_2026.0.0.0_x86_64.zip" `
    -OutFile "openvino_genai.zip" -Proxy $env:http_proxy
Expand-Archive -Path "openvino_genai.zip" -DestinationPath "." -Force
Remove-Item "openvino_genai.zip"
```

After extraction you'll have:
```
ai-pc-training/
└── openvino_genai_windows_2026.0.0.0_x86_64/
    ├── setupvars.ps1      ← environment setup script
    ├── setupvars.bat
    ├── runtime/            ← DLLs, headers, CMake configs
    ├── samples/
    ├── python/
    └── docs/
```

### Step 4: Set Up OpenVINO Environment

**Run this in every new terminal session before building or running:**

```powershell
& .\openvino_genai_windows_2026.0.0.0_x86_64\setupvars.ps1
```

Expected output:
```
[setupvars] OpenVINO environment initialized
[setupvars] OpenVINO Python environment initialized
```

### Step 5: Build the Sample

```powershell
cd samples\cpp\visual_language_chat
cmake -B build
cmake --build build --config Release
```

Build output: `build\Release\medgemma_medical_chat.exe` (~456 KB)

### Step 6: Export MedGemma Model (One-Time)

Create a Python virtual environment and export the model (~3.3 GB output):

```powershell
# From repo root
cd C:\working\ai-pc-training

# Create and activate Python venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install export dependencies
pip install --proxy $env:http_proxy optimum-intel openvino openvino-tokenizers nncf

# Login to Hugging Face (needed for gated model access)
huggingface-cli login

# Export model with INT4 quantization (~5 min)
optimum-cli export openvino -m google/medgemma-1.5-4b-it `
    --task image-text-to-text --weight-format int4 --trust-remote-code `
    medgemma-1.5-4b-it-int4
```

> **Note**: You must first accept the [MedGemma license](https://huggingface.co/google/medgemma-1.5-4b-it) on Hugging Face, or the download will fail.

Exported model directory (`medgemma-1.5-4b-it-int4/`, ~3.3 GB):
```
openvino_language_model.bin          (~2.2 GB)
openvino_vision_embeddings_model.bin (~406 MB)
openvino_text_embeddings_model.bin   (~640 MB)
+ tokenizer files, configs, XML model descriptors
```

### Step 7: Run the Demo

```powershell
# Make sure OpenVINO environment is set (Step 4)
& .\openvino_genai_windows_2026.0.0.0_x86_64\setupvars.ps1

# Run with GPU (recommended)
.\samples\cpp\visual_language_chat\build\Release\medgemma_medical_chat.exe `
    medgemma-1.5-4b-it-int4 `
    samples\cpp\visual_language_chat\chest_xray_sample.png `
    GPU

# Or with CPU (slower but always works)
.\samples\cpp\visual_language_chat\build\Release\medgemma_medical_chat.exe `
    medgemma-1.5-4b-it-int4 `
    samples\cpp\visual_language_chat\chest_xray_sample.png `
    CPU
```

### Step 8: Interact with MedGemma

Once loaded, type commands at the `Question:` prompt:

| Command | Action |
|---|---|
| `/describe` | General description of the medical image |
| `/report` | Structured radiology findings report |
| `/abnormal` | Focus on detecting abnormalities |
| `/anatomy` | Identify visible anatomical structures |
| `/help` | Show all commands |
| `/quit` | Exit |

Or type any free-form medical question about the image.

### Verified Test Output (Feb 2026)

```
Question: /describe
Using prompt: Describe this X-ray image in detail. Include any notable findings.

MedGemma: This is a chest X-ray image.

**Overall Appearance:**
*   The image shows the chest cavity, including the ribs, lungs, heart, and mediastinum.
*   The patient is likely in a normal position, with the chest X-ray taken in a standard
    anteroposterior (AP) view.

**Notable Findings:**
*   Heart Size: The heart size appears within normal limits.
*   Lung Fields: The lung fields are clear, with no obvious consolidation, effusions, or masses.
*   Mediastinum: The mediastinum appears normal in width and contour.
...
```

---

## Tested Environment

| Component | Version |
|---|---|
| OpenVINO GenAI | 2026.0.0.0 |
| Python | 3.12.0 |
| optimum-intel | 1.27.0 |
| transformers | 4.57.6 |
| torch | 2.10.0 |
| nncf | 3.0.0 |
| Visual Studio | 2022 (MSVC 19.44) |
| OS | Windows 11 |

## Source

Content verified and adapted from:
- [`jlee52tw/openvino.genai@3f3f4ca`](https://github.com/jlee52tw/openvino.genai/commit/3f3f4ca8d8d9b283e398a8992e69020f2f8b1938) — MedGemma medical image analysis sample
- [`jlee52tw/openvino.genai@a628c2f`](https://github.com/jlee52tw/openvino.genai/commit/a628c2faffda81af0a9fcfeebc71bd1cfbccdce1) — MedGemma documentation and test image

Both commits verified with OpenVINO GenAI 2026.0.0 release package.
