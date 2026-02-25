// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * MedASR Medical Speech Recognition Sample
 *
 * This sample demonstrates how to use Google's MedASR model with OpenVINO
 * for medical speech-to-text transcription tasks such as:
 * - Radiology dictation
 * - Clinical note dictation
 * - Medical terminology recognition
 *
 * Model: google/medasr (Conformer-CTC architecture, 105M parameters)
 * Input: 16kHz mono WAV → mel spectrogram (128 bins) → OpenVINO inference
 * Output: CTC-decoded medical transcription text
 *
 * Export command:
 *     python export_model.py --output medasr-openvino
 *
 * Usage:
 *     medasr_medical_asr <MODEL_DIR> <WAV_FILE> [DEVICE]
 *
 * Example:
 *     medasr_medical_asr ./medasr-openvino test_audio.wav CPU
 *
 * DISCLAIMER: MedASR is intended for research and development purposes only.
 * It should NOT be used for clinical diagnosis or patient care decisions.
 * All outputs require verification by qualified medical professionals.
 */

#include "audio_utils.hpp"

#include <openvino/openvino.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

// Load vocabulary mapping (token ID -> string) from vocab_id2token.json
std::unordered_map<int, std::string> load_vocabulary(const std::filesystem::path& model_dir) {
    std::unordered_map<int, std::string> id_to_token;

    auto vocab_path = model_dir / "vocab_id2token.json";
    if (!std::filesystem::exists(vocab_path)) {
        throw std::runtime_error("vocab_id2token.json not found in: " + model_dir.string());
    }

    std::ifstream vocab_file(vocab_path);
    nlohmann::json vocab_json;
    vocab_file >> vocab_json;

    // Format: {"0": "<epsilon>", "1": "<s>", "2": "</s>", ...}
    for (auto& [id_str, token] : vocab_json.items()) {
        id_to_token[std::stoi(id_str)] = token.get<std::string>();
    }

    std::cout << "Loaded vocabulary: " << id_to_token.size() << " tokens\n";
    return id_to_token;
}

// CTC greedy decoding: argmax → collapse repeats → remove blank tokens
// Handles SentencePiece ▁ (U+2581) as word boundary (space)
std::string ctc_greedy_decode(const float* logits, size_t time_steps, size_t vocab_size,
                               const std::unordered_map<int, std::string>& vocab,
                               int blank_id = 0) {
    // Step 1: Collapse repeats and remove blanks
    std::vector<int> collapsed;
    int prev_token = -1;
    for (size_t t = 0; t < time_steps; t++) {
        int best_id = 0;
        float best_score = logits[t * vocab_size];
        for (size_t v = 1; v < vocab_size; v++) {
            if (logits[t * vocab_size + v] > best_score) {
                best_score = logits[t * vocab_size + v];
                best_id = static_cast<int>(v);
            }
        }
        if (best_id != blank_id && best_id != prev_token) {
            collapsed.push_back(best_id);
        }
        prev_token = best_id;
    }

    // Step 2: Convert token IDs to text with SentencePiece handling
    // ▁ (U+2581, UTF-8: 0xE2 0x96 0x81) represents word boundary (space)
    std::string result;
    static const std::string sp_prefix = "\xe2\x96\x81";  // ▁ in UTF-8

    for (int id : collapsed) {
        auto it = vocab.find(id);
        if (it == vocab.end()) continue;
        const std::string& token = it->second;

        // Skip special tokens like </s>, <s>
        if (token == "</s>" || token == "<s>" || token == "<unk>") continue;

        // Handle SentencePiece word boundary prefix
        if (token.size() >= sp_prefix.size() &&
            token.substr(0, sp_prefix.size()) == sp_prefix) {
            result += " " + token.substr(sp_prefix.size());
        } else {
            result += token;
        }
    }

    // Trim leading whitespace
    size_t start = result.find_first_not_of(' ');
    if (start == std::string::npos) return "";
    return result.substr(start);
}

void print_usage(const char* program_name) {
    std::cout << "MedASR Medical Speech Recognition\n"
              << "==================================\n\n"
              << "Usage: " << program_name << " <MODEL_DIR> <WAV_FILE> [DEVICE]\n\n"
              << "Arguments:\n"
              << "  MODEL_DIR  - Path to the exported MedASR OpenVINO model directory\n"
              << "  WAV_FILE   - Path to a 16kHz mono WAV audio file\n"
              << "  DEVICE     - (Optional) Inference device: CPU, GPU, or NPU (default: CPU)\n\n"
              << "Example:\n"
              << "  " << program_name << " ./medasr-openvino test_audio.wav GPU\n\n"
              << "Export the model first:\n"
              << "  python export_model.py --output medasr-openvino\n\n"
              << "DISCLAIMER: This tool is for research purposes only.\n"
              << "Do NOT use for clinical diagnosis or patient care.\n";
}

void print_disclaimer() {
    std::cout << "\n"
              << "================================================================\n"
              << "|              IMPORTANT DISCLAIMER                            |\n"
              << "================================================================\n"
              << "| MedASR is designed for RESEARCH AND DEVELOPMENT only.        |\n"
              << "| Outputs are NOT intended for clinical diagnosis or treatment.|\n"
              << "| All results require verification by qualified professionals. |\n"
              << "================================================================\n\n";
}

int main(int argc, char* argv[]) try {
    if (argc < 3 || argc > 4) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    std::filesystem::path model_dir = argv[1];
    std::string wav_path = argv[2];
    std::string device = (argc == 4) ? argv[3] : "CPU";

    if (!std::filesystem::exists(model_dir)) {
        std::cerr << "Error: Model directory does not exist: " << model_dir << "\n";
        return EXIT_FAILURE;
    }
    if (!std::filesystem::exists(wav_path)) {
        std::cerr << "Error: WAV file does not exist: " << wav_path << "\n";
        return EXIT_FAILURE;
    }

    print_disclaimer();

    // Load vocabulary
    std::cout << "Loading vocabulary...\n";
    auto vocabulary = load_vocabulary(model_dir);

    // Load mel filterbank matrix (257 x 128)
    utils::audio::MelConfig mel_config;
    int freq_bins = mel_config.n_fft / 2 + 1;  // 257
    auto mel_filters_path = (model_dir / "mel_filters.bin").string();
    std::cout << "Loading mel filterbank...\n";
    auto mel_filters = utils::audio::load_mel_filters(mel_filters_path, freq_bins, mel_config.n_mels);

    // Read audio
    std::cout << "Reading audio: " << wav_path << "\n";
    auto raw_audio = utils::audio::read_wav(wav_path);
    double audio_duration = raw_audio.size() / 16000.0;
    std::cout << "Audio duration: " << std::fixed << std::setprecision(1)
              << audio_duration << " seconds (" << raw_audio.size() << " samples)\n";

    // Compute mel spectrogram
    std::cout << "Computing mel spectrogram...\n";
    size_t num_frames = 0;
    auto mel_features = utils::audio::compute_mel_spectrogram(raw_audio, mel_config, mel_filters, num_frames);
    std::cout << "Mel spectrogram: " << num_frames << " frames x " << mel_config.n_mels << " bins\n";

    // Initialize OpenVINO Runtime
    std::cout << "Device: " << device << "\n";
    std::cout << "Initializing OpenVINO Runtime...\n";
    ov::Core core;

    // Find the model XML file
    std::filesystem::path model_xml;
    for (auto& entry : std::filesystem::directory_iterator(model_dir)) {
        if (entry.path().extension() == ".xml") {
            model_xml = entry.path();
            break;
        }
    }
    if (model_xml.empty()) {
        std::cerr << "Error: No .xml model file found in: " << model_dir << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "Loading model: " << model_xml.filename() << "\n";
    auto model = core.read_model(model_xml);

    ov::AnyMap config;
    if (device.find("GPU") != std::string::npos) {
        config.insert({ov::cache_dir("medasr_cache")});
    }

    auto compiled_model = core.compile_model(model, device, config);
    auto infer_request = compiled_model.create_infer_request();

    // Prepare input tensors
    // input_features: [1, num_frames, 128]
    ov::Shape features_shape = {1, num_frames, static_cast<size_t>(mel_config.n_mels)};
    ov::Tensor features_tensor(ov::element::f32, features_shape, mel_features.data());

    // attention_mask: [1, num_frames] - all true (no padding), boolean type
    std::vector<uint8_t> attn_mask(num_frames, 1);
    ov::Shape mask_shape = {1, num_frames};
    ov::Tensor mask_tensor(ov::element::boolean, mask_shape, attn_mask.data());

    infer_request.set_tensor("input_features", features_tensor);
    infer_request.set_tensor("attention_mask", mask_tensor);

    // Run inference
    std::cout << "Running inference...\n";
    auto start_time = std::chrono::high_resolution_clock::now();
    infer_request.infer();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Get output and decode
    auto output_tensor = infer_request.get_output_tensor();
    auto output_shape = output_tensor.get_shape();
    std::cout << "Output shape: [" << output_shape[0] << ", " << output_shape[1]
              << ", " << output_shape[2] << "]\n";

    const float* logits = output_tensor.data<float>();
    size_t time_steps = output_shape[1];
    size_t vocab_size = output_shape[2];

    std::string transcription = ctc_greedy_decode(logits, time_steps, vocab_size, vocabulary);

    // Display results
    std::cout << "\n================================================================\n";
    std::cout << "Transcription:\n";
    std::cout << "================================================================\n";
    std::cout << transcription << "\n";
    std::cout << "================================================================\n";
    std::cout << "Inference time: " << duration.count() << " ms\n";
    std::cout << "Audio duration: " << std::fixed << std::setprecision(1)
              << audio_duration << " s\n";
    std::cout << "Real-time factor: " << std::fixed << std::setprecision(2)
              << (duration.count() / 1000.0) / audio_duration << "x\n";

    std::cout << "\nRemember: Always verify medical transcriptions with qualified professionals.\n";

    return EXIT_SUCCESS;

} catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Unknown error occurred\n";
    return EXIT_FAILURE;
}
