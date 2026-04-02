# AI PC Training — Project Plan

> **目標：** 將 [openvino_notebooks](https://github.com/openvinotoolkit/openvino_notebooks) 及相關範例轉換為
> 獨立的 CLI 專案，供 OEM/ODM 客戶在 Intel AI PC 平台上進行技術訓練。
>
> **目標平台：** Windows 11 + Intel iGPU (Arc / Iris Xe) + Intel NPU + OpenVINO
>
> **Repository：** https://github.com/jlee52tw/ai-pc-training

---

## Completed Tasks

### 1. MedGemma Medical Image Analysis (C++)

| Item | Detail |
|------|--------|
| **Source** | `jlee52tw/openvino.genai` — MedGemma VLM sample |
| **Location** | [`samples/cpp/visual_language_chat/`](samples/cpp/visual_language_chat/) |
| **Type** | C++ / OpenVINO GenAI VLMPipeline |
| **Status** | ✅ Complete — committed & pushed |

**Features:**
- Interactive medical image Q&A (`/describe`, `/report`, `/abnormal`, `/anatomy`)
- Multi-turn conversation with streaming output
- GPU and CPU inference support
- Includes sample chest X-ray test image

---

### 2. ACT — Action Chunking with Transformers (Robotics / OpenVINO)

| Item | Detail |
|------|--------|
| **Source** | `openvino_notebooks/notebooks/aloha-act/aloha-act.ipynb` |
| **Location** | [`aloha-act/`](aloha-act/) |
| **Type** | Python CLI / OpenVINO + PyTorch + MuJoCo |
| **Status** | ✅ Complete — committed & pushed |

**Deliverables:**

| File | Description |
|------|-------------|
| `main.py` | 推論 CLI — 3 步操作 (setup → convert → evaluate) |
| `train.py` | 訓練 CLI — 4 步操作 (generate → train → convert → evaluate) |
| `requirements.txt` | 鎖定版本的 Python 相依套件 |
| `README.md` | 完整繁體中文文件 (參數說明、結果分析、疑難排解) |

**Verified Results (Intel iGPU):**

| Metric | Result |
|--------|--------|
| Success Rate (pre-trained, insertion) | 90% (9/10 episodes) |
| Inference Speed | ~21 FPS |
| Model Size (OpenVINO IR) | 124.5 MB |

**Training Pipeline (train.py):**
- `generate` — MuJoCo scripted demo data (50 episodes)
- `train` — ACT model training on CPU (PyTorch, ~69 sec/epoch)
- `convert` — PyTorch → OpenVINO IR
- `evaluate` — OpenVINO inference on Intel iGPU
- Auto-patches DETR argparser and cuda calls for CPU-only training
- Supports: `sim_transfer_cube_scripted`, `sim_insertion_scripted`

---

### 3. Gemma 3 270M — LLM on CPU / GPU / NPU

| Item | Detail |
|------|--------|
| **Source** | `openvino_notebooks/notebooks/gemma3/gemma3.ipynb` + `google/gemma-3-270m` |
| **Location** | [`gemma3-270m/`](gemma3-270m/) |
| **Type** | Python CLI / OpenVINO GenAI LLMPipeline |
| **Status** | ✅ Complete — committed & pushed |

**Deliverables:**

| File | Description |
|------|-------------|
| `main.py` | CLI — 3 子命令 (export → benchmark → chat) |
| `README.md` | 完整繁體中文文件 (效能比較、NPU 量化說明) |

**Verified Results (3 devices):**

| Device | Throughput | TPOT | TTFT | Model Size |
|--------|-----------|------|------|------------|
| CPU | 83.71 tokens/s | 11.95 ms | 20.53 ms | 832 MB (FP16) |
| GPU (iGPU) | **145.10 tokens/s** | 6.89 ms | 15.11 ms | 832 MB (FP16) |
| NPU | 100.51 tokens/s | 9.95 ms | 85.96 ms | 209 MB (INT4) |

**Key Features:**
- Dual export: FP16 (CPU/GPU) + INT4-NPU (symmetric per-channel, ratio 1.0)
- Python-based benchmark_genai (embedded, no external exe needed)
- Interactive chat with streaming output
- Intel lab proxy auto-configured

---

## Next Tasks (Candidates)

Below are candidate notebooks from `openvino_notebooks` that could be converted
to standalone training projects. Priority is based on customer interest and
demo visibility.

### High Priority — Vision & Multimodal

| # | Notebook | Description | Complexity |
|---|----------|-------------|------------|
| 3 | `llm-chatbot` | LLM chatbot with OpenVINO (多模型支援) | Medium |
| 4 | `vlm-chatbot` | Vision-Language Model chatbot | Medium |
| 5 | `whisper-asr-genai` | Whisper speech recognition (GenAI API) | Low |
| 6 | `text-to-image-genai` | Stable Diffusion text-to-image (GenAI API) | Medium |
| 7 | `depth-anything` | Monocular depth estimation | Low |

### Medium Priority — Object Detection & Segmentation

| # | Notebook | Description | Complexity |
|---|----------|-------------|------------|
| 8 | `yolov12-optimization` | YOLOv12 object detection + quantization | Medium |
| 9 | `sam2-image-segmentation` | Segment Anything 2 | Medium |
| 10 | `florence2` | Florence-2 multimodal vision model | Medium |
| 11 | `grounded-segment-anything` | Grounded SAM (text prompt → segmentation) | High |

### Medium Priority — Audio & Speech

| # | Notebook | Description | Complexity |
|---|----------|-------------|------------|
| 12 | `kokoro` | Kokoro TTS (text-to-speech) | Low |
| 13 | `bark-text-to-audio` | Bark text-to-audio generation | Low |
| 14 | `openvoice2-and-melotts` | Voice cloning + TTS | Medium |

### Lower Priority — Specialized

| # | Notebook | Description | Complexity |
|---|----------|-------------|------------|
| 15 | `deepseek-r1` | DeepSeek reasoning model | Medium |
| 16 | `qwen2.5-vl` | Qwen 2.5 vision-language model | Medium |
| 17 | `llm-rag-langchain` | RAG with LangChain + OpenVINO | High |
| 18 | `handwritten-ocr` | Handwritten text OCR | Low |
| 19 | `image-classification-quantization` | INT8 quantization tutorial | Low |

---

## Project Conventions

| Convention | Rule |
|------------|------|
| **Target OS** | Windows 11 |
| **Python** | 3.10 – 3.12, venv-based |
| **Structure** | One subfolder per task (e.g., `aloha-act/`) |
| **CLI style** | Single script with subcommands (`argparse`) |
| **Device default** | `--device GPU` (Intel iGPU) |
| **Proxy** | Intel lab proxy auto-configured, `--no-proxy` flag to disable |
| **Documentation** | README.md in Traditional Chinese (繁體中文) |
| **Dependencies** | Pinned versions in `requirements.txt` per subfolder |
| **Large files** | Excluded via `.gitignore` (model weights, training data) |

---

## Timeline

| Date | Milestone |
|------|-----------|
| 2026-03 | ✅ MedGemma C++ sample |
| 2026-04 | ✅ ACT robotics demo (main.py + train.py) |
| 2026-04 | ✅ Gemma 3 270M — CPU / GPU / NPU benchmark |
| 2026-04+ | Next task TBD |

---

*Last updated: 2026-04-02*
