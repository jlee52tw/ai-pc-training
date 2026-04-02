#!/usr/bin/env python
"""
Gemma 3 270M — OpenVINO 推論 & 效能測試 (CPU / GPU / NPU)
==========================================================
將 Google Gemma-3-270m 匯出為 OpenVINO IR，並在 Intel CPU、iGPU、NPU 上
執行效能基準測試與互動式文字生成。

子命令
------
  export     匯出 HuggingFace 模型為 OpenVINO IR (支援 CPU/GPU 與 NPU 兩種格式)
  benchmark  執行 benchmark_genai 效能測試
  chat       互動式文字生成 (LLMPipeline)

完整流程
--------
  python main.py export                          # 匯出模型 (CPU/GPU + NPU)
  python main.py benchmark --device CPU          # CPU 基準測試
  python main.py benchmark --device GPU          # iGPU 基準測試
  python main.py benchmark --device NPU          # NPU 基準測試
  python main.py chat --device GPU               # 互動式聊天
"""

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_DIR = pathlib.Path(__file__).resolve().parent
MODEL_ID = "google/gemma-3-270m"

# Export directories (relative to PROJECT_DIR)
EXPORT_DIR_DEFAULT = PROJECT_DIR / "gemma-3-270"          # CPU / GPU (dynamic)
EXPORT_DIR_NPU     = PROJECT_DIR / "gemma-3-270-npu"      # NPU (INT4-NPU, static)

# Intel lab proxy
INTEL_HTTP_PROXY = "http://proxy-dmz.intel.com:912"
INTEL_NO_PROXY   = ".intel.com,intel.com,localhost,127.0.0.1"

# Default benchmark parameters
DEFAULT_PROMPT     = "What is artificial intelligence?"
DEFAULT_MAX_TOKENS = 50
DEFAULT_NUM_ITER   = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _banner(msg: str) -> None:
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")


def _step(msg: str) -> None:
    print(f"[STEP] {msg}")


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _error(msg: str) -> None:
    print(f"[ERROR] {msg}")


def _set_proxy(env: dict) -> dict:
    """Inject Intel lab proxy variables."""
    for key in ("http_proxy", "HTTP_PROXY"):
        env[key] = INTEL_HTTP_PROXY
    for key in ("https_proxy", "HTTPS_PROXY"):
        env[key] = INTEL_HTTP_PROXY
    for key in ("no_proxy", "NO_PROXY"):
        env[key] = INTEL_NO_PROXY
    return env


def _make_env(proxy: bool = True) -> dict:
    """Return os.environ copy with optional proxy."""
    env = os.environ.copy()
    if proxy:
        _set_proxy(env)
    return env


def _get_model_dir(device: str) -> pathlib.Path:
    """Return the correct export directory based on device."""
    if device.upper() == "NPU":
        return EXPORT_DIR_NPU
    return EXPORT_DIR_DEFAULT


def _check_exported(model_dir: pathlib.Path) -> bool:
    """Check if model has been exported."""
    return (model_dir / "openvino_model.xml").exists() or \
           (model_dir / "openvino_model.bin").exists()


# ---------------------------------------------------------------------------
# Subcommand: export
# ---------------------------------------------------------------------------
def cmd_export(args):
    """匯出 HuggingFace 模型為 OpenVINO IR。"""
    _banner(f"匯出模型 — {MODEL_ID}")

    env = _make_env(proxy=not args.no_proxy)

    # --- Export 1: Default (CPU/GPU) — dynamic shapes ---
    if not args.npu_only:
        _export_default(env, force=args.force)

    # --- Export 2: NPU — INT4-NPU config ---
    if not args.default_only:
        _export_npu(env, force=args.force)

    _banner("匯出完成 ✓")
    _info("下一步 → python main.py benchmark --device CPU")
    _info("       → python main.py benchmark --device GPU")
    _info("       → python main.py benchmark --device NPU")


def _export_default(env: dict, force: bool = False):
    """Export for CPU/GPU (default FP16, dynamic shapes)."""
    _step("匯出 CPU/GPU 版本 (動態 shape, FP16)")

    if _check_exported(EXPORT_DIR_DEFAULT) and not force:
        _info(f"模型已存在: {EXPORT_DIR_DEFAULT}")
        _info("使用 --force 重新匯出。")
        return

    cmd = [
        "optimum-cli", "export", "openvino",
        "-m", MODEL_ID,
        str(EXPORT_DIR_DEFAULT),
    ]

    _info(f"執行: optimum-cli export openvino -m {MODEL_ID} {EXPORT_DIR_DEFAULT.name}")
    _info("下載模型與轉換中（首次需要下載約 1 GB）…")
    result = subprocess.run(cmd, env=env, text=True)

    if result.returncode != 0:
        _error(f"匯出失敗 (exit code {result.returncode})")
        sys.exit(result.returncode)

    _verify_export(EXPORT_DIR_DEFAULT, "CPU/GPU")


def _export_npu(env: dict, force: bool = False):
    """Export for NPU — INT4-NPU config (symmetric, per-channel, ratio 1.0)."""
    _step("匯出 NPU 版本 (INT4-NPU, symmetric per-channel)")

    if _check_exported(EXPORT_DIR_NPU) and not force:
        _info(f"模型已存在: {EXPORT_DIR_NPU}")
        _info("使用 --force 重新匯出。")
        return

    cmd = [
        "optimum-cli", "export", "openvino",
        "-m", MODEL_ID,
        "--weight-format", "int4",
        "--sym",
        "--group-size", "-1",
        "--ratio", "1.0",
        str(EXPORT_DIR_NPU),
    ]

    _info(f"INT4-NPU 配置: sym=True, group_size=-1, ratio=1.0")
    _info("轉換中…")
    result = subprocess.run(cmd, env=env, text=True)

    if result.returncode != 0:
        _error(f"匯出失敗 (exit code {result.returncode})")
        sys.exit(result.returncode)

    _verify_export(EXPORT_DIR_NPU, "NPU")


def _verify_export(model_dir: pathlib.Path, label: str):
    """Print exported model file sizes."""
    _info(f"{label} 匯出檔案:")
    for f in sorted(model_dir.glob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            if size_mb > 0.01:
                _info(f"  {f.name} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Subcommand: benchmark
# ---------------------------------------------------------------------------
def cmd_benchmark(args):
    """執行 benchmark_genai 效能測試。"""
    device = args.device.upper()
    _banner(f"效能基準測試 — {MODEL_ID} on {device}")

    model_dir = _get_model_dir(device)
    if not _check_exported(model_dir):
        _error(f"模型未匯出: {model_dir}")
        _info("請先執行 'python main.py export'")
        sys.exit(1)

    prompt = args.prompt
    max_tokens = args.max_tokens
    num_iter = args.num_iter

    _step(f"裝置: {device}")
    _step(f"模型: {model_dir}")
    _info(f"提示: \"{prompt}\"")
    _info(f"最大生成 tokens: {max_tokens}, 迭代次數: {num_iter}")

    # Use Python-based benchmark (embedded, same as benchmark_genai.py)
    _run_python_benchmark(model_dir, device, prompt, max_tokens, num_iter)

    _banner("基準測試完成 ✓")


def _run_python_benchmark(model_dir, device, prompt, max_tokens, num_iter):
    """Run benchmark using openvino_genai Python API (same as benchmark_genai.py)."""
    try:
        import openvino_genai as ov_genai
        from openvino import get_version as ov_version
    except ImportError:
        _error("需要安裝 openvino-genai: pip install openvino-genai")
        sys.exit(1)

    print(f"\nOpenVINO Runtime")
    print(f"    Version : {ov_version()}")
    print(f"    GenAI   : {ov_genai.__version__}")

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = max_tokens
    config.apply_chat_template = False

    _info(f"載入模型 → {device} …")
    load_start = time.perf_counter()

    if device == "NPU":
        pipe = ov_genai.LLMPipeline(str(model_dir), device)
    else:
        import sys as _sys
        scheduler_config = ov_genai.SchedulerConfig()
        scheduler_config.enable_prefix_caching = False
        scheduler_config.max_num_batched_tokens = _sys.maxsize
        pipe = ov_genai.LLMPipeline(str(model_dir), device, scheduler_config=scheduler_config)

    load_time_ms = (time.perf_counter() - load_start) * 1000
    _info(f"模型載入完成 ({load_time_ms:.0f} ms)")

    # Tokenize to get prompt size
    input_data = pipe.get_tokenizer().encode([prompt])
    prompt_token_size = input_data.input_ids.get_shape()[1]
    print(f"\nPrompt token size: {prompt_token_size}")

    # Warmup
    _info("Warmup (1 iteration) …")
    pipe.generate([prompt], config)

    # Benchmark iterations
    _info(f"執行 {num_iter} 次迭代 …")
    res = pipe.generate([prompt], config)
    perf_metrics = res.perf_metrics
    for _ in range(num_iter - 1):
        res = pipe.generate([prompt], config)
        perf_metrics += res.perf_metrics

    # Print results
    print(f"\nOutput token size: {res.perf_metrics.get_num_generated_tokens()}")
    print(f"Load time: {load_time_ms:.2f} ms")
    print(f"Generate time: {perf_metrics.get_generate_duration().mean:.2f}"
          f" ± {perf_metrics.get_generate_duration().std:.2f} ms")
    print(f"Tokenization time: {perf_metrics.get_tokenization_duration().mean:.2f}"
          f" ± {perf_metrics.get_tokenization_duration().std:.2f} ms")
    print(f"Detokenization time: {perf_metrics.get_detokenization_duration().mean:.2f}"
          f" ± {perf_metrics.get_detokenization_duration().std:.2f} ms")
    print(f"TTFT: {perf_metrics.get_ttft().mean:.2f}"
          f" ± {perf_metrics.get_ttft().std:.2f} ms")
    print(f"TPOT: {perf_metrics.get_tpot().mean:.2f}"
          f" ± {perf_metrics.get_tpot().std:.2f} ms/token")
    print(f"Throughput: {perf_metrics.get_throughput().mean:.2f}"
          f" ± {perf_metrics.get_throughput().std:.2f} tokens/s")


# ---------------------------------------------------------------------------
# Subcommand: chat
# ---------------------------------------------------------------------------
def cmd_chat(args):
    """互動式文字生成。"""
    device = args.device.upper()
    _banner(f"互動式聊天 — {MODEL_ID} on {device}")

    model_dir = _get_model_dir(device)
    if not _check_exported(model_dir):
        _error(f"模型未匯出: {model_dir}")
        _info("請先執行 'python main.py export'")
        sys.exit(1)

    try:
        import openvino_genai as ov_genai
    except ImportError:
        _error("需要安裝 openvino-genai: pip install openvino-genai")
        sys.exit(1)

    _info(f"載入模型 → {device} …")
    if device == "NPU":
        pipe = ov_genai.LLMPipeline(str(model_dir), device)
    else:
        import sys as _sys
        scheduler_config = ov_genai.SchedulerConfig()
        scheduler_config.enable_prefix_caching = False
        scheduler_config.max_num_batched_tokens = _sys.maxsize
        pipe = ov_genai.LLMPipeline(str(model_dir), device, scheduler_config=scheduler_config)

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = args.max_tokens
    config.apply_chat_template = False

    _info("模型已就緒。輸入文字開始生成 (輸入 'quit' 離開)。\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            break

        # Streaming output
        def streamer(subword):
            print(subword, end="", flush=True)
            return False

        pipe.generate(user_input, config, streamer)
        print("\n")

    _info("再見！")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Gemma 3 270M — OpenVINO 匯出 / 基準測試 / 聊天",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--no-proxy", action="store_true",
        help="停用 Intel lab proxy (proxy-dmz.intel.com:912)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # --- export ---
    p_export = sub.add_parser("export", help="匯出模型為 OpenVINO IR")
    p_export.add_argument(
        "--force", action="store_true",
        help="強制重新匯出（覆蓋已存在的模型）",
    )
    p_export.add_argument(
        "--default-only", action="store_true",
        help="只匯出 CPU/GPU 版本（跳過 NPU）",
    )
    p_export.add_argument(
        "--npu-only", action="store_true",
        help="只匯出 NPU 版本（跳過 CPU/GPU）",
    )
    p_export.set_defaults(func=cmd_export)

    # --- benchmark ---
    p_bench = sub.add_parser("benchmark", help="效能基準測試 (benchmark_genai)")
    p_bench.add_argument(
        "--device", default="CPU", choices=["CPU", "GPU", "NPU"],
        help="推論裝置 (預設: CPU)",
    )
    p_bench.add_argument(
        "--prompt", default=DEFAULT_PROMPT,
        help=f"測試提示詞 (預設: \"{DEFAULT_PROMPT}\")",
    )
    p_bench.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help=f"最大生成 tokens 數 (預設: {DEFAULT_MAX_TOKENS})",
    )
    p_bench.add_argument(
        "--num-iter", type=int, default=DEFAULT_NUM_ITER,
        help=f"測試迭代次數 (預設: {DEFAULT_NUM_ITER})",
    )
    p_bench.set_defaults(func=cmd_benchmark)

    # --- chat ---
    p_chat = sub.add_parser("chat", help="互動式文字生成")
    p_chat.add_argument(
        "--device", default="GPU", choices=["CPU", "GPU", "NPU"],
        help="推論裝置 (預設: GPU)",
    )
    p_chat.add_argument(
        "--max-tokens", type=int, default=100,
        help="最大生成 tokens 數 (預設: 100)",
    )
    p_chat.set_defaults(func=cmd_chat)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
