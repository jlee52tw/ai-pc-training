#!/usr/bin/env python
"""
ACT 自訂訓練腳本 — 生成資料 → 訓練 → 轉換 → 評估
=====================================================
讓 OEM/ODM 客戶可以參考此腳本，學習如何訓練自己的 ACT 模型。

子命令
------
  generate   在 MuJoCo 模擬環境中生成腳本示範資料 (scripted demo)
  train      用生成的示範資料訓練 ACT 模型
  convert    將訓練好的 PyTorch checkpoint 轉換為 OpenVINO IR
  evaluate   用 OpenVINO 在 Intel iGPU 上評估模型

完整流程 (以 sim_transfer_cube_scripted 為例)
---------------------------------------------
  python train.py generate --task sim_transfer_cube_scripted
  python train.py train    --task sim_transfer_cube_scripted --epochs 2000
  python train.py convert  --task sim_transfer_cube_scripted
  python train.py evaluate --task sim_transfer_cube_scripted --device GPU

支援的模擬任務
--------------
  sim_transfer_cube_scripted   方塊搬移 (腳本示範, 4 相機)
  sim_insertion_scripted       插銷任務 (腳本示範, 4 相機)
"""

import argparse
import os
import pathlib
import shutil
import subprocess
import sys

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_DIR = pathlib.Path(__file__).resolve().parent
TOP_DIR_NAME = "edge-ai-suites"
SPARSE_PATH = "robotics-ai-suite/pipelines/act-sample"

# Intel lab proxy
INTEL_HTTP_PROXY = "http://proxy-dmz.intel.com:912"
INTEL_NO_PROXY = ".intel.com,intel.com,localhost,127.0.0.1"

# Supported sim tasks and their camera counts
SUPPORTED_TASKS = {
    "sim_transfer_cube_scripted": {"cameras": 4, "desc": "方塊搬移 (腳本示範)"},
    "sim_insertion_scripted":     {"cameras": 4, "desc": "插銷任務 (腳本示範)"},
}

# Default ACT hyperparameters (same as original paper)
DEFAULT_HP = {
    "kl_weight": 10,
    "chunk_size": 100,
    "hidden_dim": 512,
    "dim_feedforward": 3200,
    "batch_size": 8,
    "lr": 1e-5,
    "seed": 0,
}


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


def _make_env(act_dir: pathlib.Path | None = None, proxy: bool = True) -> dict:
    """Return os.environ copy with optional PYTHONPATH & proxy."""
    env = os.environ.copy()
    # Use non-interactive matplotlib backend to avoid GUI hangs on Windows
    env["MPLBACKEND"] = "Agg"
    if proxy:
        _set_proxy(env)
    if act_dir:
        env["PYTHONPATH"] = str(act_dir) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _get_act_dir() -> pathlib.Path:
    """Return the resolved path to the ACT source code."""
    act_dir = PROJECT_DIR / TOP_DIR_NAME / SPARSE_PATH / "act"
    if not act_dir.exists():
        _error(f"ACT 目錄不存在: {act_dir}")
        _info("請先執行 'python main.py setup' 建立環境。")
        sys.exit(1)
    return act_dir


def _get_data_dir(task: str) -> pathlib.Path:
    """Return the dataset directory for a task."""
    return PROJECT_DIR / "data" / task


def _get_ckpt_dir(task: str) -> pathlib.Path:
    """Return the checkpoint directory for a task."""
    return PROJECT_DIR / "checkpoints" / task


def _patch_constants(act_dir: pathlib.Path, data_base: pathlib.Path):
    """
    Patch constants.py DATA_DIR to point to our data directory.
    This is needed so imitate_episodes.py can find the training data.
    """
    constants_file = act_dir / "constants.py"
    content = constants_file.read_text(encoding="utf-8")

    # Replace the placeholder DATA_DIR with our actual path
    data_dir_str = str(data_base).replace("\\", "/")
    new_content = content.replace(
        "DATA_DIR = '<put your data dir here>'",
        f"DATA_DIR = r'{data_dir_str}'"
    )
    if new_content != content:
        constants_file.write_text(new_content, encoding="utf-8")
        _info(f"已更新 constants.py DATA_DIR → {data_dir_str}")
    else:
        _info("constants.py DATA_DIR 已設定完成。")


def _patch_for_cpu_training(act_dir: pathlib.Path):
    """
    Patch ACT source files for CPU-only training (no CUDA).

    1. detr/main.py
       - parse_args() → parse_known_args()[0]   (ignore unknown args like --device)
       - model.cuda() → removed                 (keep model on CPU)
    2. imitate_episodes.py
       - .cuda() → .cpu() in forward_pass / train_bc
    """
    # --- Patch detr/main.py ---
    detr_main = act_dir / "detr" / "main.py"
    content = detr_main.read_text(encoding="utf-8")
    changed = False

    if "args = parser.parse_args()" in content:
        content = content.replace(
            "args = parser.parse_args()",
            "args, _ = parser.parse_known_args()",
        )
        changed = True

    if "    model.cuda()" in content:
        content = content.replace(
            "    model.cuda()",
            "    # model.cuda()  # patched: CPU-only training",
        )
        changed = True

    if changed:
        detr_main.write_text(content, encoding="utf-8")
        _info("已修補 detr/main.py (parse_known_args + 移除 cuda)")
    else:
        _info("detr/main.py 已修補完成。")

    # --- Patch imitate_episodes.py ---
    ie_file = act_dir / "imitate_episodes.py"
    content = ie_file.read_text(encoding="utf-8")
    changed = False

    # forward_pass: data tensors .cuda() → .cpu()
    old_fwd = (
        "image_data, qpos_data, action_data, is_pad = "
        "image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()"
    )
    new_fwd = (
        "image_data, qpos_data, action_data, is_pad = "
        "image_data.cpu(), qpos_data.cpu(), action_data.cpu(), is_pad.cpu()"
    )
    if old_fwd in content:
        content = content.replace(old_fwd, new_fwd)
        changed = True

    # train_bc: policy.cuda() → policy.cpu()
    if "policy.cuda()" in content:
        content = content.replace("policy.cuda()", "policy.cpu()")
        changed = True

    if changed:
        ie_file.write_text(content, encoding="utf-8")
        _info("已修補 imitate_episodes.py (cuda → cpu)")
    else:
        _info("imitate_episodes.py 已修補完成。")


def _validate_task(task: str):
    """Ensure the task name is supported."""
    if task not in SUPPORTED_TASKS:
        _error(f"不支援的任務: {task}")
        _info(f"支援的任務: {', '.join(SUPPORTED_TASKS.keys())}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Subcommand: generate
# ---------------------------------------------------------------------------
def cmd_generate(args):
    """在 MuJoCo 模擬環境中生成腳本示範資料。"""
    task = args.task
    num_episodes = args.episodes
    _validate_task(task)
    _banner(f"生成示範資料 — {task} ({num_episodes} episodes)")

    act_dir = _get_act_dir()
    data_dir = _get_data_dir(task)
    env = _make_env(act_dir=act_dir, proxy=not args.no_proxy)

    # Patch constants.py to set DATA_DIR
    _patch_constants(act_dir, PROJECT_DIR / "data")

    script = act_dir / "record_sim_episodes.py"
    if not script.exists():
        _error(f"record_sim_episodes.py 不存在: {script}")
        sys.exit(1)

    data_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(script),
        "--task_name", task,
        "--dataset_dir", str(data_dir),
        "--num_episodes", str(num_episodes),
    ]

    _step(f"工作目錄: {act_dir}")
    _step(f"輸出目錄: {data_dir}")
    _info(f"生成 {num_episodes} 個 episode 的示範資料 …")

    proc = subprocess.Popen(
        cmd, cwd=str(act_dir), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:
        print(line, end="")
    proc.wait()

    if proc.returncode != 0:
        _error(f"程序退出碼: {proc.returncode}")
        sys.exit(proc.returncode)

    # Verify
    hdf5_files = list(data_dir.glob("*.hdf5"))
    _info(f"已生成 {len(hdf5_files)} 個 .hdf5 檔案於 {data_dir}")

    _banner("資料生成完成 ✓")
    _info(f"下一步 → python train.py train --task {task}")


# ---------------------------------------------------------------------------
# Subcommand: train
# ---------------------------------------------------------------------------
def cmd_train(args):
    """用示範資料訓練 ACT 模型。"""
    task = args.task
    epochs = args.epochs
    _validate_task(task)
    _banner(f"訓練 ACT 模型 — {task} ({epochs} epochs)")

    act_dir = _get_act_dir()
    data_dir = _get_data_dir(task)
    ckpt_dir = _get_ckpt_dir(task)
    env = _make_env(act_dir=act_dir, proxy=not args.no_proxy)

    # Patch constants.py and source files for CPU training
    _patch_constants(act_dir, PROJECT_DIR / "data")
    _patch_for_cpu_training(act_dir)

    # Verify training data exists
    hdf5_files = list(data_dir.glob("*.hdf5"))
    if not hdf5_files:
        _error(f"找不到訓練資料: {data_dir}")
        _info(f"請先執行 'python train.py generate --task {task}'")
        sys.exit(1)
    _info(f"找到 {len(hdf5_files)} 個訓練 episode")

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    script = act_dir / "imitate_episodes.py"
    cmd = [
        sys.executable, str(script),
        "--task_name", task,
        "--ckpt_dir", str(ckpt_dir),
        "--policy_class", "ACT",
        "--kl_weight", str(DEFAULT_HP["kl_weight"]),
        "--chunk_size", str(DEFAULT_HP["chunk_size"]),
        "--hidden_dim", str(DEFAULT_HP["hidden_dim"]),
        "--dim_feedforward", str(DEFAULT_HP["dim_feedforward"]),
        "--batch_size", str(DEFAULT_HP["batch_size"]),
        "--lr", str(DEFAULT_HP["lr"]),
        "--seed", str(DEFAULT_HP["seed"]),
        "--num_epochs", str(epochs),
        "--device", "CPU",  # Training uses PyTorch on CPU (required arg)
    ]

    _step(f"工作目錄: {act_dir}")
    _step(f"訓練資料: {data_dir}")
    _step(f"模型輸出: {ckpt_dir}")
    _info("開始訓練（CPU / PyTorch）… 這可能需要數小時。")
    _info(f"超參數: chunk_size={DEFAULT_HP['chunk_size']}, hidden_dim={DEFAULT_HP['hidden_dim']}, "
          f"kl_weight={DEFAULT_HP['kl_weight']}, dim_feedforward={DEFAULT_HP['dim_feedforward']}")

    proc = subprocess.Popen(
        cmd, cwd=str(act_dir), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:
        print(line, end="")
    proc.wait()

    if proc.returncode != 0:
        _error(f"程序退出碼: {proc.returncode}")
        sys.exit(proc.returncode)

    # Verify output
    ckpt_file = ckpt_dir / "policy_best.ckpt"
    if ckpt_file.exists():
        size_mb = ckpt_file.stat().st_size / (1024 * 1024)
        _info(f"模型已儲存: {ckpt_file} ({size_mb:.1f} MB)")
    else:
        _warn("找不到 policy_best.ckpt — 請檢查訓練輸出。")

    _banner("訓練完成 ✓")
    _info(f"下一步 → python train.py convert --task {task}")


# ---------------------------------------------------------------------------
# Subcommand: convert
# ---------------------------------------------------------------------------
def cmd_convert(args):
    """將訓練好的 PyTorch checkpoint 轉換為 OpenVINO IR。"""
    task = args.task
    _validate_task(task)
    _banner(f"模型轉換 — {task} (PyTorch → OpenVINO IR)")

    act_dir = _get_act_dir()
    ckpt_dir = _get_ckpt_dir(task)
    ckpt_path = ckpt_dir / "policy_best.ckpt"
    env = _make_env(act_dir=act_dir, proxy=not args.no_proxy)

    if not ckpt_path.exists():
        _error(f"模型權重不存在: {ckpt_path}")
        _info(f"請先執行 'python train.py train --task {task}'")
        sys.exit(1)

    script = act_dir / "ov_convert.py"
    camera_num = SUPPORTED_TASKS[task]["cameras"]

    cmd = [
        sys.executable, str(script),
        "--ckpt_path", str(ckpt_path),
        "--height", "480",
        "--weight", "640",
        "--camera_num", str(camera_num),
        "--chunk_size", str(DEFAULT_HP["chunk_size"]),
    ]

    _step(f"工作目錄: {act_dir}")
    _step(f"輸入 checkpoint: {ckpt_path}")
    _info(f"相機數量: {camera_num}")

    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(act_dir), env=env)
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr)
        _error(f"轉換失敗 (exit code {res.returncode})")
        sys.exit(res.returncode)

    # Verify
    xml_file = ckpt_dir / "policy_best.xml"
    bin_file = ckpt_dir / "policy_best.bin"
    if xml_file.exists() and bin_file.exists():
        _info(f"OpenVINO IR 檔案:")
        _info(f"  {xml_file.name} ({xml_file.stat().st_size / (1024*1024):.1f} MB)")
        _info(f"  {bin_file.name} ({bin_file.stat().st_size / (1024*1024):.1f} MB)")
    else:
        _warn("找不到 .xml/.bin — 請檢查轉換輸出。")

    _banner("轉換完成 ✓")
    _info(f"下一步 → python train.py evaluate --task {task} --device GPU")


# ---------------------------------------------------------------------------
# Subcommand: evaluate
# ---------------------------------------------------------------------------
def cmd_evaluate(args):
    """在 OpenVINO 上評估訓練好的模型。"""
    task = args.task
    device = args.device
    _validate_task(task)
    _banner(f"評估模型 — {task} (裝置: {device})")

    act_dir = _get_act_dir()
    ckpt_dir = _get_ckpt_dir(task)
    env = _make_env(act_dir=act_dir, proxy=not args.no_proxy)

    # Patch constants.py (needed for evaluation sim environment)
    _patch_constants(act_dir, PROJECT_DIR / "data")

    script = act_dir / "imitate_episodes.py"
    if not (ckpt_dir / "policy_best.xml").exists():
        _error(f"OpenVINO IR 不存在: {ckpt_dir / 'policy_best.xml'}")
        _info(f"請先執行 'python train.py convert --task {task}'")
        sys.exit(1)

    cmd = [
        sys.executable, str(script),
        "--task_name", task,
        "--ckpt_dir", str(ckpt_dir),
        "--policy_class", "ACT",
        "--kl_weight", str(DEFAULT_HP["kl_weight"]),
        "--chunk_size", str(DEFAULT_HP["chunk_size"]),
        "--hidden_dim", str(DEFAULT_HP["hidden_dim"]),
        "--dim_feedforward", str(DEFAULT_HP["dim_feedforward"]),
        "--batch_size", str(DEFAULT_HP["batch_size"]),
        "--lr", str(DEFAULT_HP["lr"]),
        "--seed", str(DEFAULT_HP["seed"]),
        "--num_epochs", "2000",
        "--device", device,
        "--eval",
    ]

    _step(f"工作目錄: {act_dir}")
    _step(f"模型目錄: {ckpt_dir}")
    _info("執行 10 個 rollout episode … 約需 10-20 分鐘。")

    proc = subprocess.Popen(
        cmd, cwd=str(act_dir), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:
        print(line, end="")
    proc.wait()

    if proc.returncode != 0:
        _error(f"程序退出碼: {proc.returncode}")
        sys.exit(proc.returncode)

    _banner("評估完成 ✓")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="ACT 自訂訓練工具 — 生成資料 / 訓練 / 轉換 / 評估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--no-proxy", action="store_true",
        help="停用 Intel lab proxy (proxy-dmz.intel.com:912)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # --- generate ---
    p_gen = sub.add_parser("generate", help="生成模擬示範資料")
    p_gen.add_argument(
        "--task", default="sim_transfer_cube_scripted",
        choices=list(SUPPORTED_TASKS.keys()),
        help="任務名稱 (預設: sim_transfer_cube_scripted)",
    )
    p_gen.add_argument(
        "--episodes", type=int, default=50,
        help="生成的 episode 數量 (預設: 50)",
    )
    p_gen.set_defaults(func=cmd_generate)

    # --- train ---
    p_train = sub.add_parser("train", help="訓練 ACT 模型")
    p_train.add_argument(
        "--task", default="sim_transfer_cube_scripted",
        choices=list(SUPPORTED_TASKS.keys()),
        help="任務名稱 (預設: sim_transfer_cube_scripted)",
    )
    p_train.add_argument(
        "--epochs", type=int, default=2000,
        help="訓練輪數 (預設: 2000)",
    )
    p_train.set_defaults(func=cmd_train)

    # --- convert ---
    p_conv = sub.add_parser("convert", help="轉換模型 → OpenVINO IR")
    p_conv.add_argument(
        "--task", default="sim_transfer_cube_scripted",
        choices=list(SUPPORTED_TASKS.keys()),
        help="任務名稱 (預設: sim_transfer_cube_scripted)",
    )
    p_conv.set_defaults(func=cmd_convert)

    # --- evaluate ---
    p_eval = sub.add_parser("evaluate", help="評估模型 (OpenVINO)")
    p_eval.add_argument(
        "--task", default="sim_transfer_cube_scripted",
        choices=list(SUPPORTED_TASKS.keys()),
        help="任務名稱 (預設: sim_transfer_cube_scripted)",
    )
    p_eval.add_argument(
        "--device", default="GPU", choices=["GPU", "CPU", "AUTO"],
        help="OpenVINO 推論裝置 (預設: GPU = Intel iGPU)",
    )
    p_eval.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
