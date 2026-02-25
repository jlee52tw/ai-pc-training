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
        from transformers import AutoModelForCTC, AutoConfig, AutoProcessor
        from safetensors.torch import load_file
        import torch
        import numpy as np
        import openvino as ov
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install transformers torch openvino safetensors")
        sys.exit(1)

    model_id = "google/medasr"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load processor (for feature extraction config and vocab)
    print(f"Loading processor from: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)

    # Workaround: from_pretrained hangs on Windows, load manually
    print(f"Loading model weights manually...")
    config = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCTC.from_config(config)
    from huggingface_hub import hf_hub_download
    safetensors_path = hf_hub_download(model_id, "model.safetensors")
    model.load_state_dict(load_file(safetensors_path))
    model.eval()
    print(f"Model loaded: {type(model).__name__} ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")

    # Save processor files (tokenizer, preprocessor config, etc.)
    processor.save_pretrained(str(output_path))
    print("Saved processor files")

    # Save feature extraction config for C++ (needed for mel spectrogram)
    fe_config = {
        "sampling_rate": processor.feature_extractor.sampling_rate,
        "n_fft": processor.feature_extractor.n_fft,
        "hop_length": processor.feature_extractor.hop_length,
        "win_length": processor.feature_extractor.win_length,
        "feature_size": processor.feature_extractor.feature_size,
    }
    with open(output_path / "feature_extraction_config.json", "w") as f:
        json.dump(fe_config, f, indent=2)
    print(f"Feature extraction config: {fe_config}")

    # Save vocabulary as id->raw_token mapping for C++ CTC decoding
    # Use convert_ids_to_tokens to preserve SentencePiece ▁ prefix
    tokenizer = processor.tokenizer
    vocab_map = {}
    for token_id in range(config.vocab_size):
        raw_token = tokenizer.convert_ids_to_tokens(token_id)
        if raw_token is not None:
            vocab_map[str(token_id)] = raw_token
    with open(output_path / "vocab_id2token.json", "w") as f:
        json.dump(vocab_map, f, indent=2, ensure_ascii=False)
    print(f"Vocabulary saved: {len(vocab_map)} tokens")

    # Workaround for transformers bug: _torch_extract_fbank_features
    # called with extra 'center' arg that method doesn't accept
    orig_method = type(processor.feature_extractor)._torch_extract_fbank_features
    def patched_extract(self, waveform, device="cpu", center=False):
        return orig_method(self, waveform, device)
    type(processor.feature_extractor)._torch_extract_fbank_features = patched_extract

    # Create dummy mel features using the processor to get correct shape
    dummy_audio = np.random.randn(16000 * 5).astype(np.float32)  # 5 seconds
    inputs = processor(dummy_audio, sampling_rate=16000, return_tensors="pt")
    print(f"Input keys: {list(inputs.keys())}")
    for k, v in inputs.items():
        if hasattr(v, "shape"):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

    # Test forward pass
    print("Testing forward pass...")
    with torch.no_grad():
        outputs = model(**inputs)
        print(f"Output logits shape: {outputs.logits.shape}")

    # Export model to OpenVINO IR
    print("Converting model to OpenVINO IR...")
    example_input = {k: v for k, v in inputs.items() if hasattr(v, "shape")}

    # Determine dynamic shapes based on input
    input_shapes = []
    for k, v in example_input.items():
        shape = list(v.shape)
        # Make batch and sequence dims dynamic
        dyn_shape = [-1] + [-1] * (len(shape) - 1)
        input_shapes.append(ov.PartialShape(dyn_shape))

    ov_model = ov.convert_model(
        model,
        example_input=example_input,
        input=input_shapes,
    )

    model_xml = str(output_path / "medasr_model.xml")
    ov.save_model(ov_model, model_xml)
    print(f"Model saved to: {model_xml}")

    print(f"\nExport complete! Model directory: {output_dir}")
    print("Files created:")
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
