# ACT — Action Chunking with Transformers (OpenVINO on Intel iGPU)

A standalone training demo that runs **ACT imitation-learning** inference on
**Intel iGPU** (Arc / Iris Xe) using **OpenVINO**.

This project converts the
[aloha-act notebook](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/aloha-act)
into a single Python CLI for easy step-by-step OEM/ODM tech training.

---

## What This Demo Does

ACT (Action Chunking with Transformers) is an imitation-learning architecture
that uses a Conditional VAE with Transformers to replicate expert robot
behavior. This demo:

1. Downloads a **pre-trained ACT checkpoint** for a simulated peg-insertion
   task (4 cameras).
2. Converts the PyTorch model to **OpenVINO IR** format.
3. Runs **10 evaluation rollout episodes** using the OpenVINO runtime on your
   Intel iGPU, reporting success rate, FPS, and reward metrics.

| Angle Camera | Left Wrist | Right Wrist | Top Camera |
|:---:|:---:|:---:|:---:|
| ![cameras](https://github.com/open-edge-platform/edge-ai-suites/raw/main/robotics-ai-suite/pipelines/act-sample/README.assets/act-sim-cameras.png) |

![Peg Insertion Demo](https://github.com/open-edge-platform/edge-ai-suites/raw/main/robotics-ai-suite/pipelines/act-sample/README.assets/act-sim-insertion-demo.gif)

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| **OS** | Windows 11 |
| **Python** | 3.10 – 3.12 |
| **Git** | 2.30+ (with sparse-checkout support) |
| **GPU** | Intel iGPU — Arc / Iris Xe (for `--device GPU`) |
| **Driver** | Latest Intel GPU driver from [intel.com/drivers](https://www.intel.com/content/www/us/en/download-center/home.html) |
| **Network** | Intel lab proxy access (auto-configured) |

---

## Quick Start (3 Steps)

### Step 0 — Create a Virtual Environment

```powershell
cd C:\working\ai-pc-training\aloha-act

python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 1 — Setup: Download Weights, Clone Repos, Install Deps

```powershell
python main.py setup
```

This command will:
- Configure Intel lab proxy (`proxy-dmz.intel.com:912`)
- Download the pre-trained checkpoint (`sim_insertion_scripted.zip`)
- Sparse-clone the [edge-ai-suites](https://github.com/open-edge-platform/edge-ai-suites) repo
  and initialise the ACT submodule
- Apply OpenVINO and IPEX patches
- Install all Python dependencies from `requirements.txt`
- Install the DETR package from source

> **Note:** If you are outside the Intel lab network, add `--no-proxy`:
> ```powershell
> python main.py setup --no-proxy
> ```

Expected output (last lines):
```
============================================================
  Setup Complete ✓
============================================================
[INFO] Next step → python main.py convert
```

### Step 2 — Convert: PyTorch → OpenVINO IR

```powershell
python main.py convert
```

This runs `ov_convert.py` to convert the PyTorch `.ckpt` file to OpenVINO IR
(`.xml` + `.bin`).

Expected output:
```
============================================================
  Conversion Complete ✓
============================================================
[INFO] Next step → python main.py evaluate --device GPU
```

### Step 3 — Evaluate: Run on Intel iGPU

```powershell
python main.py evaluate --device GPU
```

This runs 10 rollout episodes of the simulated peg-insertion task using
OpenVINO inference on your Intel iGPU.

| Flag | Value | Description |
|------|-------|-------------|
| `--device` | `GPU` (default) | Intel iGPU via OpenVINO |
| `--device` | `CPU` | Fall back to CPU |
| `--device` | `AUTO` | Let OpenVINO auto-select the best device |

> **Note:** Evaluation takes **10 – 20 minutes** depending on your hardware.

---

## Understanding the Output

### Per-Episode Metrics

| Metric | Meaning |
|--------|---------|
| `episode_return` | Total cumulative reward for the episode |
| `episode_highest_reward` | Max reward achieved in a single step |
| `env_max_reward` | Maximum possible reward per step (4) |
| `Success` | Whether the episode completed successfully |
| `Avg fps` | Inference speed (frames per second) |

### Overall Results

| Metric | Meaning |
|--------|---------|
| **Success rate** | % of episodes that completed the peg insertion |
| **Average return** | Mean cumulative reward across all episodes |
| **Reward ≥ 0** | Basic interaction achieved |
| **Reward ≥ 1–3** | Partial task completion |
| **Reward ≥ 4** | Full task completion (successful insertion) |

A perfect run = **100% success rate** with all episodes at reward level 4.

---

## Project Structure

```
aloha-act/
├── main.py              # Single CLI entry point (setup / convert / evaluate)
├── requirements.txt     # Pinned Python dependencies
├── README.md            # This file
│
├── sim_insertion_scripted/          # (created by setup)
│   └── four_camera/
│       └── policy_best.ckpt        # Pre-trained weights
│
└── edge-ai-suites/                 # (created by setup — sparse checkout)
    └── robotics-ai-suite/
        └── pipelines/
            └── act-sample/
                ├── act/             # ACT source code (patched)
                │   ├── ov_convert.py
                │   ├── imitate_episodes.py
                │   └── detr/
                └── patches/
                    ├── ov/          # OpenVINO patches
                    └── ipex/        # IPEX patches
```

---

## Troubleshooting

### Proxy issues
If downloads fail or git clone times out, verify the proxy is reachable:
```powershell
$env:http_proxy = "http://proxy-dmz.intel.com:912"
$env:https_proxy = "http://proxy-dmz.intel.com:912"
curl -I https://github.com
```

### GPU not detected by OpenVINO
Make sure the latest Intel GPU driver is installed. Verify with:
```python
import openvino as ov
print(ov.Core().available_devices)
# Should include 'GPU'
```

### MuJoCo rendering errors on Windows
MuJoCo uses native OpenGL on Windows. If you see rendering errors:
```powershell
$env:MUJOCO_GL = "egl"
python main.py evaluate --device GPU
```
Or fall back to software rendering:
```powershell
$env:MUJOCO_GL = "osmesa"
```

### Patch apply warnings
Warnings like `patch already applied` during setup are harmless and can be
ignored.

### Out of memory
Reduce `--batch_size` in `main.py` → `cmd_evaluate()` if your system runs
out of GPU memory (unlikely on iGPU with this model size).

---

## References

- [ACT Paper — arXiv:2304.13705](https://arxiv.org/pdf/2304.13705)
- [Original ACT Repo](https://github.com/tonyzhaozh/act)
- [Intel Edge AI Suites](https://github.com/open-edge-platform/edge-ai-suites)
- [OpenVINO Notebooks — aloha-act](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/aloha-act)
- [OpenVINO Documentation](https://docs.openvino.ai/)

---

## License

This demo project is provided for OEM/ODM tech-training purposes.
The underlying ACT model and code are subject to their respective licenses.
See the [OpenVINO Notebooks LICENSE](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/LICENSE).
