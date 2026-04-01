#!/usr/bin/env python
"""
ACT (Action Chunking with Transformers) — Standalone Training Demo
===================================================================
Single entry-point for OEM/ODM tech-training on Intel iGPU platforms.

Subcommands
-----------
  setup     Download weights, clone repos, apply patches, install deps
  convert   Convert PyTorch checkpoint → OpenVINO IR
  evaluate  Run 10 rollout episodes on the selected OpenVINO device

Usage
-----
  python main.py setup
  python main.py convert
  python main.py evaluate --device GPU

Environment
-----------
Designed for Windows 11 with Intel iGPU (Arc / Iris Xe).
Intel lab proxy is configured automatically (override with --no-proxy).
"""

import argparse
import importlib
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import zipfile

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_DIR = pathlib.Path(__file__).resolve().parent

# Pre-trained weights
ZIP_URL = "https://eci.intel.com/embodied-sdk-docs/_downloads/sim_insertion_scripted.zip"
CKPT_REL = pathlib.Path("sim_insertion_scripted/four_camera/policy_best.ckpt")

# Edge AI Suites (sparse checkout)
TOP_REPO_URL = "https://github.com/open-edge-platform/edge-ai-suites.git"
TOP_DIR_NAME = "edge-ai-suites"
SPARSE_PATH = "robotics-ai-suite/pipelines/act-sample"

# Original ACT repo (fallback)
TONYZ_ACT_URL = "https://github.com/tonyzhaozh/act.git"
TONYZ_ACT_COMMIT = "742c753c0d4a5d87076c8f69e5628c79a8cc5488"

# Patches to skip
SKIP_PATCHES = {"0006-add-ros2-node-and-use-fixed-cube-pose.patch"}

# Intel lab proxy
INTEL_HTTP_PROXY = "http://proxy-dmz.intel.com:912"
INTEL_NO_PROXY = ".intel.com,intel.com,localhost,127.0.0.1"


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


def _run(cmd: list, cwd=None, env=None, check=True, capture=False):
    """Run a subprocess and optionally stream or capture output."""
    _info(f"Running: {' '.join(str(c) for c in cmd)}")
    if capture:
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=env)
        if check and res.returncode != 0:
            _error(res.stderr)
            raise subprocess.CalledProcessError(res.returncode, cmd)
        return res
    else:
        subprocess.check_call(cmd, cwd=cwd, env=env)


def _set_proxy(env: dict) -> dict:
    """Inject Intel lab proxy variables into *env*."""
    env["http_proxy"] = INTEL_HTTP_PROXY
    env["https_proxy"] = INTEL_HTTP_PROXY
    env["no_proxy"] = INTEL_NO_PROXY
    env["HTTP_PROXY"] = INTEL_HTTP_PROXY
    env["HTTPS_PROXY"] = INTEL_HTTP_PROXY
    env["NO_PROXY"] = INTEL_NO_PROXY
    return env


def _make_env(act_dir: pathlib.Path | None = None, proxy: bool = True) -> dict:
    """Return an os.environ copy with optional PYTHONPATH & proxy."""
    env = os.environ.copy()
    if proxy:
        _set_proxy(env)
    if act_dir:
        env["PYTHONPATH"] = str(act_dir) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _resolve_paths():
    """Return resolved key paths relative to PROJECT_DIR."""
    top_dir = PROJECT_DIR / TOP_DIR_NAME
    repo_dir = top_dir / SPARSE_PATH
    act_dir = repo_dir / "act"
    detr_dir = act_dir / "detr"
    ckpt_path = PROJECT_DIR / CKPT_REL
    ckpt_dir = ckpt_path.parent
    return top_dir, repo_dir, act_dir, detr_dir, ckpt_path, ckpt_dir


# ---------------------------------------------------------------------------
# Subcommand: setup
# ---------------------------------------------------------------------------
def cmd_setup(args):
    """Download weights, clone repos, patch, install deps."""
    _banner("ACT Setup — Environment Preparation")
    top_dir, repo_dir, act_dir, detr_dir, ckpt_path, _ = _resolve_paths()
    env = _make_env(proxy=not args.no_proxy)

    # ---- 0. Ensure 'requests' is available (needed for download) ----
    try:
        import requests
    except ModuleNotFoundError:
        _step("Installing 'requests' (needed for download) …")
        _run([sys.executable, "-m", "pip", "install", "requests"], env=env)
        import requests

    # ---- 1. Download pre-trained weights ----
    if ckpt_path.exists():
        _info(f"Checkpoint already exists: {ckpt_path}")
    else:
        _step("Downloading pre-trained weights …")

        zip_path = PROJECT_DIR / "sim_insertion_scripted.zip"
        resp = requests.get(ZIP_URL, stream=True, proxies={
            "http": env.get("http_proxy", ""),
            "https": env.get("https_proxy", ""),
        })
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        _info(f"Downloaded → {zip_path}")

        _step("Extracting archive …")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(PROJECT_DIR)
        zip_path.unlink()
        _info("Extraction complete, zip removed.")

    # ---- 2. Sparse-clone edge-ai-suites ----
    if not top_dir.exists():
        _step(f"Sparse-cloning {TOP_REPO_URL} …")
        clone_env = env.copy()
        clone_env["GIT_LFS_SKIP_SMUDGE"] = "1"
        _run(["git", "clone", "--filter=blob:none", "--sparse",
              TOP_REPO_URL, str(top_dir)], env=clone_env)
    else:
        _info(f"Repo directory already exists: {top_dir}")

    _step("Setting sparse-checkout path …")
    _run(["git", "-C", str(top_dir), "sparse-checkout", "init", "--cone"], env=env)
    _run(["git", "-C", str(top_dir), "sparse-checkout", "set", SPARSE_PATH], env=env)
    _info(f"Checked out: {repo_dir}")

    # ---- 3. Initialize ACT submodule (with fallback) ----
    submodule_path = str(pathlib.PurePosixPath(SPARSE_PATH) / "act")
    try:
        _step("Initializing ACT submodule …")
        _run(["git", "-C", str(top_dir), "submodule", "update",
              "--init", "--depth", "1", submodule_path], env=env)
        _info("ACT submodule initialized.")
    except subprocess.CalledProcessError:
        _warn("Submodule init failed — falling back to direct clone.")

    if not act_dir.exists() or not any(act_dir.iterdir()):
        act_dir.mkdir(parents=True, exist_ok=True)
        _step(f"Cloning ACT repo → {act_dir}")
        _run(["git", "clone", TONYZ_ACT_URL, str(act_dir)], env=env)
        _run(["git", "-C", str(act_dir), "checkout", TONYZ_ACT_COMMIT], env=env)
        _info(f"Checked out commit {TONYZ_ACT_COMMIT}")

    # ---- 4. Create __init__.py files for DETR ----
    for sub in [detr_dir, detr_dir / "models"]:
        init_file = sub / "__init__.py"
        if sub.exists() and not init_file.exists():
            _step(f"Creating {init_file}")
            init_file.write_text("# auto-generated\n")

    # ---- 5. Apply patches ----
    patches_base = repo_dir / "patches"
    for subdir in ("ov", "ipex"):
        patch_dir = patches_base / subdir
        if not patch_dir.exists():
            _warn(f"Patches directory not found: {patch_dir}")
            continue
        _step(f"Applying patches from {subdir}/ …")
        for pf in sorted(patch_dir.glob("*.patch")):
            if pf.name in SKIP_PATCHES:
                _info(f"Skipping {pf.name}")
                continue
            res = subprocess.run(
                ["git", "-C", str(act_dir), "apply", "--ignore-whitespace",
                 str(pf.resolve())],
                capture_output=True, text=True, env=env,
            )
            if res.returncode == 0:
                _info(f"Applied: {pf.name}")
            else:
                _warn(f"Patch {pf.name} failed (may already be applied): {res.stderr.strip()}")

    # ---- 6. Install Python dependencies ----
    _step("Upgrading pip / setuptools / wheel …")
    _run([sys.executable, "-m", "pip", "install", "-U",
          "pip", "setuptools", "wheel"], env=env)

    _step("Installing requirements …")
    req_file = PROJECT_DIR / "requirements.txt"
    _run([sys.executable, "-m", "pip", "install", "-r", str(req_file)], env=env)

    # ---- 7. Install DETR from source ----
    if detr_dir.exists():
        _step(f"Installing DETR from {detr_dir} …")
        _run([sys.executable, "-m", "pip", "install", "."],
             cwd=str(detr_dir), env=env)
        _info("DETR installed.")
    else:
        _error(f"DETR directory not found: {detr_dir}")

    _banner("Setup Complete ✓")
    _info("Next step → python main.py convert")


# ---------------------------------------------------------------------------
# Subcommand: convert
# ---------------------------------------------------------------------------
def cmd_convert(args):
    """Convert PyTorch checkpoint to OpenVINO IR."""
    _banner("Model Conversion — PyTorch → OpenVINO IR")
    _, _, act_dir, _, ckpt_path, _ = _resolve_paths()
    env = _make_env(act_dir=act_dir, proxy=not args.no_proxy)

    script = act_dir / "ov_convert.py"
    if not script.exists():
        _error(f"ov_convert.py not found: {script}")
        sys.exit(1)
    if not ckpt_path.exists():
        _error(f"Checkpoint not found: {ckpt_path}")
        _info("Run 'python main.py setup' first.")
        sys.exit(1)

    cmd = [
        sys.executable, str(script),
        "--ckpt_path", str(ckpt_path),
        "--height", "480",
        "--weight", "640",
        "--camera_num", "4",
        "--chunk_size", "100",
    ]

    _step(f"Working directory: {act_dir}")
    _step(" ".join(str(c) for c in cmd))

    res = _run(cmd, cwd=str(act_dir), env=env, capture=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)

    # Verify output files exist
    ov_dir = ckpt_path.parent
    xml_files = list(ov_dir.glob("*.xml"))
    bin_files = list(ov_dir.glob("*.bin"))
    if xml_files and bin_files:
        _info(f"OpenVINO IR files generated in {ov_dir}:")
        for f in sorted(xml_files + bin_files):
            _info(f"  {f.name}")
    else:
        # Also check inside act_dir for generated files
        xml_files2 = list(act_dir.glob("**/*.xml"))
        bin_files2 = list(act_dir.glob("**/*.bin"))
        if xml_files2 and bin_files2:
            _info("OpenVINO IR files generated:")
            for f in sorted(xml_files2 + bin_files2):
                _info(f"  {f.relative_to(act_dir)}")
        else:
            _warn("Could not verify OpenVINO IR output — check console output above.")

    _banner("Conversion Complete ✓")
    _info("Next step → python main.py evaluate --device GPU")


# ---------------------------------------------------------------------------
# Subcommand: evaluate
# ---------------------------------------------------------------------------
def cmd_evaluate(args):
    """Run 10 evaluation episodes on the chosen OpenVINO device."""
    device = args.device
    _banner(f"Evaluation — 10 rollout episodes on {device}")
    _, _, act_dir, _, _, ckpt_dir = _resolve_paths()
    env = _make_env(act_dir=act_dir, proxy=not args.no_proxy)

    script = act_dir / "imitate_episodes.py"
    if not script.exists():
        _error(f"imitate_episodes.py not found: {script}")
        sys.exit(1)
    if not ckpt_dir.exists():
        _error(f"Checkpoint directory not found: {ckpt_dir}")
        sys.exit(1)

    cmd = [
        sys.executable, str(script),
        "--task_name", "sim_insertion_scripted",
        "--ckpt_dir", str(ckpt_dir),
        "--policy_class", "ACT",
        "--kl_weight", "10",
        "--chunk_size", "100",
        "--hidden_dim", "512",
        "--batch_size", "8",
        "--dim_feedforward", "3200",
        "--num_epochs", "2000",
        "--lr", "1e-5",
        "--seed", "0",
        "--device", device,
        "--eval",
    ]

    _step(f"Working directory: {act_dir}")
    _step(" ".join(str(c) for c in cmd))
    _info("This may take 10–20 minutes depending on hardware …")

    # On Windows MuJoCo uses native OpenGL — no DISPLAY check needed.
    # Stream output in real-time so trainees can watch progress.
    proc = subprocess.Popen(
        cmd, cwd=str(act_dir), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:
        print(line, end="")
    proc.wait()

    if proc.returncode != 0:
        _error(f"Process exited with code {proc.returncode}")
        sys.exit(proc.returncode)

    _banner("Evaluation Complete ✓")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="ACT Imitation-Learning Demo — Intel iGPU Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--no-proxy", action="store_true",
        help="Disable Intel lab proxy (proxy-dmz.intel.com:912)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # setup
    p_setup = sub.add_parser("setup", help="Download weights, clone repos, install deps")
    p_setup.set_defaults(func=cmd_setup)

    # convert
    p_convert = sub.add_parser("convert", help="Convert PyTorch model → OpenVINO IR")
    p_convert.set_defaults(func=cmd_convert)

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Run 10 evaluation episodes")
    p_eval.add_argument(
        "--device", default="GPU", choices=["GPU", "CPU", "AUTO"],
        help="OpenVINO device for inference (default: GPU = Intel iGPU)",
    )
    p_eval.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
