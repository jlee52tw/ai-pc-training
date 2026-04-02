# Gemma 3 270M — OpenVINO 推論 & 效能測試 (CPU / GPU / NPU)

在 Intel AI PC 上以 **OpenVINO** 執行 Google **Gemma-3-270m** 語言模型，
支援 **CPU、iGPU、NPU** 三種加速裝置的匯出、效能基準測試與互動式文字生成。

---

## What This Demo Does

Gemma 3 270M 是 Google 發布的輕量 Transformer 語言模型（270M 參數），適合在
邊緣裝置上進行快速推論。此專案展示：

1. **匯出模型**：用 `optimum-cli` 將 HuggingFace 模型轉換為 OpenVINO IR
   - CPU/GPU 版本：FP16 動態 shape
   - NPU 版本：INT4 對稱量化（per-channel, ratio 1.0）
2. **效能基準測試**：在 CPU / iGPU / NPU 上執行 `benchmark_genai` 風格的效能測試
3. **互動式聊天**：使用 `LLMPipeline` 進行即時文字生成

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| **OS** | Windows 11 |
| **Python** | 3.10 – 3.12 |
| **GPU** | Intel iGPU — Arc / Iris Xe (for `--device GPU`) |
| **NPU** | Intel AI Boost NPU (for `--device NPU`) |
| **Driver** | 最新 Intel GPU + NPU 驅動 ([intel.com/drivers](https://www.intel.com/content/www/us/en/download-center/home.html)) |
| **Network** | Intel lab proxy access (自動設定) |

### Python 套件

| Package | Version |
|---------|---------|
| `openvino` | ≥ 2025.3.0 |
| `openvino-genai` | ≥ 2025.3.0 |
| `optimum-intel` | ≥ 1.25.0 |
| `transformers` | ≥ 4.45.0 |
| `nncf` | ≥ 2.18.0 |

```powershell
pip install openvino openvino-genai optimum-intel[openvino] nncf
```

---

## Quick Start

### Step 1 — 匯出模型

```powershell
cd C:\working\ai-pc-training\gemma3-270m
python main.py export
```

這會自動匯出兩種版本：

| 版本 | 目錄 | 格式 | 大小 | 用途 |
|------|------|------|------|------|
| CPU/GPU | `gemma-3-270/` | FP16 | ~832 MB | CPU 與 iGPU 推論 |
| NPU | `gemma-3-270-npu/` | INT4-NPU | ~209 MB | NPU 推論 |

> **首次執行** 需從 HuggingFace 下載模型（約 1 GB），後續使用快取。
>
> 不在 Intel lab 網路時加 `--no-proxy`：
> ```powershell
> python main.py export --no-proxy
> ```

### Step 2 — 效能基準測試

```powershell
# CPU 測試
python main.py benchmark --device CPU

# Intel iGPU 測試
python main.py benchmark --device GPU

# Intel NPU 測試
python main.py benchmark --device NPU
```

### Step 3 — 互動式聊天

```powershell
python main.py chat --device GPU
```

輸入文字後模型會即時生成回應（streaming output）。輸入 `quit` 離開。

---

## 效能基準測試結果

### 測試環境

| Item | Detail |
|------|--------|
| CPU | Intel Core Ultra (具體型號依測試機而定) |
| iGPU | Intel Arc / Iris Xe |
| NPU | Intel AI Boost |
| OpenVINO | 2025.4.1 |
| OpenVINO GenAI | 2025.4.1.0 |
| Prompt | "What is artificial intelligence?" (6 tokens) |
| Max tokens | 50 |
| Iterations | 5 |

### 結果比較

| 指標 | CPU | GPU (iGPU) | NPU |
|------|-----|------------|-----|
| **Load time** | 1,466 ms | 4,598 ms | 16,728 ms |
| **Generate time** | 606.27 ± 14.00 ms | 353.25 ± 7.40 ms | 574.11 ± 2.65 ms |
| **TTFT** | 20.53 ± 1.33 ms | 15.11 ± 1.82 ms | 85.96 ± 1.40 ms |
| **TPOT** | 11.95 ± 1.01 ms/token | 6.89 ± 1.06 ms/token | 9.95 ± 0.58 ms/token |
| **Throughput** | **83.71 ± 7.05** tokens/s | **145.10 ± 22.38** tokens/s | **100.51 ± 5.83** tokens/s |

### 指標說明

| 指標 | 說明 |
|------|------|
| **Load time** | 模型載入時間（含編譯）。NPU 編譯較慢但只需一次 |
| **Generate time** | 完整生成 50 tokens 所需時間 |
| **TTFT** | Time To First Token — 第一個 token 出現的延遲 |
| **TPOT** | Time Per Output Token — 每個 token 的生成時間 |
| **Throughput** | 每秒生成的 token 數（越高越好） |

### 結果分析

| 裝置 | 特性 | 適用場景 |
|------|------|---------|
| **GPU** | 最高吞吐量 (145 tokens/s)、最低 TPOT | 即時聊天、高效能推論 |
| **NPU** | 良好吞吐量 (100 tokens/s)、最低變異 | 低功耗持續推論、背景 AI 任務 |
| **CPU** | 最快載入 (1.5s)、穩定表現 | 無 GPU/NPU 的環境、開發測試 |

> **重點：** GPU 推論速度最快（145 tokens/s），NPU 則以較低功耗達到 100 tokens/s，
> 適合需要長時間運行的 AI 應用。CPU 載入最快，適合快速原型開發。

---

## 命令參數詳解

### `export` — 模型匯出

```powershell
python main.py export [--force] [--default-only] [--npu-only] [--no-proxy]
```

| 參數 | 說明 |
|------|------|
| `--force` | 強制重新匯出（覆蓋已存在的模型） |
| `--default-only` | 只匯出 CPU/GPU 版本（跳過 NPU） |
| `--npu-only` | 只匯出 NPU 版本（跳過 CPU/GPU） |
| `--no-proxy` | 停用 Intel lab proxy |

**NPU INT4 量化配置：**

| 參數 | 值 | 說明 |
|------|-----|------|
| `--weight-format` | `int4` | INT4 權重量化 |
| `--sym` | *(旗標)* | **對稱量化** — NPU 硬體最佳化為對稱權重 |
| `--group-size` | `-1` | **Per-channel 量化** — 逐通道量化（非分組），NPU 架構較適合 |
| `--ratio` | `1.0` | **100% INT4** — 所有層都量化為 INT4，無 INT4/INT8 混合 |

> **為何 NPU 需要特殊量化？**
>
> Intel NPU 的矩陣運算單元針對對稱 INT4 權重進行了硬體加速。使用 per-channel
> 量化（`group_size=-1`）比 group-wise 更適合 NPU 的計算模式，可以獲得更好的
> 推論效能。

### `benchmark` — 效能測試

```powershell
python main.py benchmark --device <CPU|GPU|NPU> [--prompt "..."] [--max-tokens N] [--num-iter N]
```

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--device` | `CPU` | 推論裝置 |
| `--prompt` | `"What is artificial intelligence?"` | 測試提示詞 |
| `--max-tokens` | `50` | 最大生成 tokens 數 |
| `--num-iter` | `5` | 測試迭代次數 |

> **注意：** NPU 會自動使用 `gemma-3-270-npu/` 目錄的 INT4 模型，CPU/GPU 使用
> `gemma-3-270/` 目錄的 FP16 模型。

### `chat` — 互動式聊天

```powershell
python main.py chat --device <CPU|GPU|NPU> [--max-tokens N]
```

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--device` | `GPU` | 推論裝置 |
| `--max-tokens` | `100` | 每次回應的最大 tokens 數 |

---

## 模型架構

```
Gemma 3 270M (Text-only LLM)
├── 270M 參數 (18 層 Transformer)
├── 隱藏維度: 1536
├── 注意力頭: 4 (GQA)
├── 詞彙表: 262,144
│
├── CPU/GPU 版本 (FP16)
│   └── gemma-3-270/openvino_model.xml + .bin (832 MB)
│
└── NPU 版本 (INT4-NPU)
    └── gemma-3-270-npu/openvino_model.xml + .bin (209 MB)
```

---

## 專案結構

```
gemma3-270m/
├── main.py              # CLI: export / benchmark / chat
├── README.md            # This file
│
├── gemma-3-270/         # CPU/GPU model (FP16, created by export)
│   ├── openvino_model.xml
│   ├── openvino_model.bin
│   ├── openvino_tokenizer.*
│   └── ...
│
└── gemma-3-270-npu/     # NPU model (INT4-NPU, created by export)
    ├── openvino_model.xml
    ├── openvino_model.bin
    ├── openvino_tokenizer.*
    └── ...
```

---

## Troubleshooting

### Proxy issues
```powershell
$env:http_proxy = "http://proxy-dmz.intel.com:912"
$env:https_proxy = "http://proxy-dmz.intel.com:912"
$env:no_proxy = ".intel.com,intel.com,localhost,127.0.0.1"
```

### 確認可用裝置
```python
import openvino as ov
print(ov.Core().available_devices)
# 應包含 ['CPU', 'GPU', 'NPU']
```

### NPU 載入時間較長
NPU 首次載入需要編譯模型（~17 秒），這是正常現象。後續推論速度正常。

### HuggingFace 存取
Gemma 3 270M 需要接受 Google 的使用條款。若下載失敗：
```powershell
pip install huggingface_hub
huggingface-cli login
```

### 安裝 hf_xet 加速下載
```powershell
pip install hf_xet
```

---

## References

- [Google Gemma 3 on HuggingFace](https://huggingface.co/google/gemma-3-270m)
- [OpenVINO GenAI Documentation](https://docs.openvino.ai/2025/openvino-workflow-generative/)
- [OpenVINO on NPU](https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html)
- [Optimum-Intel Export Guide](https://huggingface.co/docs/optimum/intel/openvino/export)
- [OpenVINO Notebooks — gemma3](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/gemma3)

---

## License

This demo project is provided for OEM/ODM tech-training purposes.
The Gemma model is subject to Google's [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
