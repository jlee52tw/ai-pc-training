#!/usr/bin/env python3
# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Export Google MedASR model to OpenVINO IR format.

This script downloads the MedASR model from Hugging Face and exports it
to OpenVINO Intermediate Representation (IR) for use with the C++ sample.

Prerequisites:
    pip install optimum-intel openvino openvino-tokenizers nncf
    pip install git+https://github.com/huggingface/transformers.git@65dc261512cbdb1ee72b88ae5b222f2605aad8e5

Usage:
    python export_model.py --output medasr-openvino
    python export_model.py --output medasr-openvino --device GPU
"""

import argparse
import json
import os
import sys
from pathlib import Path


def export_with_optimum(output_dir: str):
    """Export MedASR using optimum-cli export."""
    import subprocess

    print("Exporting MedASR model to OpenVINO IR format...")
    print(f"Output directory: {output_dir}")

    cmd = [
        sys.executable, "-m", "optimum.exporters.openvino",
        "--model", "google/medasr",
        "--task", "ctc",
        "--trust-remote-code",
        output_dir,
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print("\noptimum export failed. Trying alternative export method...")
        export_manual(output_dir)


def export_manual(output_dir: str):
    """Manual export using transformers + openvino converter."""
    try:
        from transformers import AutoModelForCTC, AutoProcessor
        import torch
        import openvino as ov
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install transformers torch openvino")
        sys.exit(1)

    model_id = "google/medasr"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading MedASR model from Hugging Face: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCTC.from_pretrained(model_id, trust_remote_code=True)
    model.eval()

    # Save processor files (vocab.json, etc.)
    processor.save_pretrained(str(output_path))
    print("Saved processor files (vocab.json, tokenizer config, etc.)")

    # Export model to OpenVINO IR
    print("Converting model to OpenVINO IR...")

    # Create dummy input matching expected audio shape
    dummy_input = torch.randn(1, 16000 * 10)  # 10 seconds of audio
    dummy_input_values = {"input_values": dummy_input}

    ov_model = ov.convert_model(
        model,
        example_input=dummy_input_values,
        input=[ov.PartialShape([-1, -1])],  # dynamic batch and length
    )

    model_xml = str(output_path / "medasr_model.xml")
    ov.save_model(ov_model, model_xml)
    print(f"Model saved to: {model_xml}")

    # Verify vocab.json exists
    vocab_json = output_path / "vocab.json"
    if vocab_json.exists():
        with open(vocab_json, "r") as f:
            vocab = json.load(f)
        print(f"Vocabulary size: {len(vocab)} tokens")
    else:
        print("WARNING: vocab.json not found - CTC decoding may not work correctly")

    print(f"\nExport complete! Model directory: {output_dir}")
    print(f"Files created:")
    for f in sorted(output_path.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name:<40} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Export Google MedASR to OpenVINO IR format"
    )
    parser.add_argument(
        "--output", "-o",
        default="medasr-openvino",
        help="Output directory for exported model (default: medasr-openvino)"
    )
    parser.add_argument(
        "--method",
        choices=["optimum", "manual"],
        default="manual",
        help="Export method: 'optimum' uses optimum-cli, 'manual' uses direct conversion (default: manual)"
    )
    args = parser.parse_args()

    # Check HuggingFace login
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user = api.whoami()
        print(f"Logged in as: {user.get('name', user.get('fullname', 'unknown'))}")
    except Exception:
        print("WARNING: Not logged in to Hugging Face.")
        print("MedASR is a gated model. Login with: huggingface-cli login")
        print("And accept the license at: https://huggingface.co/google/medasr\n")

    if args.method == "optimum":
        export_with_optimum(args.output)
    else:
        export_manual(args.output)


if __name__ == "__main__":
    main()
