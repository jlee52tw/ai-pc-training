# AI PC Training — April 2026

Training materials and samples for AI on PC development using Intel OpenVINO.

> **Project Plan:** See [PLAN.md](PLAN.md) for full roadmap and candidate tasks.

## Contents

### ACT — Action Chunking with Transformers (Robotics)

A standalone Python CLI that runs **ACT imitation-learning** inference and training
on **Intel iGPU** using **OpenVINO**, converted from the
[aloha-act notebook](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/aloha-act).

- **Inference CLI**: [`aloha-act/main.py`](aloha-act/) — setup / convert / evaluate (pre-trained model)
- **Training CLI**: [`aloha-act/train.py`](aloha-act/) — generate / train / convert / evaluate (custom model)
- **Documentation**: [`aloha-act/README.md`](aloha-act/README.md) (繁體中文)
- **Results**: 90% success rate, ~21 FPS on Intel iGPU

### Gemma 3 270M — LLM on CPU / GPU / NPU

Export and benchmark Google **Gemma-3-270m** on all 3 Intel AI PC accelerators
using **OpenVINO GenAI LLMPipeline**.

- **CLI**: [`gemma3-270m/main.py`](gemma3-270m/) — export / benchmark / chat
- **Documentation**: [`gemma3-270m/README.md`](gemma3-270m/README.md) (繁體中文)
- **Results**: CPU 84 tokens/s, **GPU 145 tokens/s**, NPU 101 tokens/s

### MedGemma Medical Image Analysis (OpenVINO GenAI)

A C++ sample demonstrating medical image analysis using Google's [MedGemma](https://huggingface.co/google/medgemma-1.5-4b-it) vision-language model with the OpenVINO GenAI VLMPipeline API.

- **Sample code**: [`samples/cpp/visual_language_chat/`](samples/cpp/visual_language_chat/)
- **Documentation**: [`samples/cpp/visual_language_chat/README.md`](samples/cpp/visual_language_chat/README.md)
- **Detailed guide**: [`samples/cpp/visual_language_chat/MEDGEMMA_SAMPLE.md`](samples/cpp/visual_language_chat/MEDGEMMA_SAMPLE.md)

Features:
- Interactive medical image Q&A with predefined prompts (`/describe`, `/report`, `/abnormal`, `/anatomy`)
- Multi-turn conversation with streaming output
- GPU and CPU inference support
- Includes sample chest X-ray test image (Public Domain)

### Quick Setup

```powershell
# Set proxy (if behind Intel proxy)
$env:http_proxy = "http://proxy-dmz.intel.com:912"
$env:https_proxy = "http://proxy-dmz.intel.com:912"

# Download and extract OpenVINO GenAI package
Invoke-WebRequest -Uri "https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2026.0/windows/openvino_genai_windows_2026.0.0.0_x86_64.zip" -OutFile "openvino_genai.zip" -Proxy $env:http_proxy
Expand-Archive -Path "openvino_genai.zip" -DestinationPath "." -Force
Remove-Item "openvino_genai.zip"

# Set up OpenVINO environment
& .\openvino_genai_windows_2026.0.0.0_x86_64\setupvars.ps1
```

### Prerequisites

- [OpenVINO GenAI 2026.0.0+](https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2026.0/windows/) (downloaded above)
- CMake 3.16+
- Visual Studio 2022 (Windows)
- [Hugging Face account](https://huggingface.co/) with MedGemma license accepted

## Source

Content verified and adapted from:
- [`jlee52tw/openvino.genai@3f3f4ca`](https://github.com/jlee52tw/openvino.genai/commit/3f3f4ca8d8d9b283e398a8992e69020f2f8b1938) — MedGemma medical image analysis sample
- [`jlee52tw/openvino.genai@a628c2f`](https://github.com/jlee52tw/openvino.genai/commit/a628c2faffda81af0a9fcfeebc71bd1cfbccdce1) — MedGemma documentation and test image

Both commits verified with the latest OpenVINO GenAI package.
