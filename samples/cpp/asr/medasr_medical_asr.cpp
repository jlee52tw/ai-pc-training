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
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

// Load vocabulary mapping (token ID -> string) from vocab.json
std::unordered_map<int, std::string> load_vocabulary(const std::filesystem::path& model_dir) {
    std::unordered_map<int, std::string> id_to_token;

    auto vocab_path = model_dir / "vocab.json";
    if (!std::filesystem::exists(vocab_path)) {
        throw std::runtime_error("vocab.json not found in model directory: " + model_dir.string());
    }

    std::ifstream vocab_file(vocab_path);
    nlohmann::json vocab_json;
    vocab_file >> vocab_json;

    // vocab.json maps token_string -> token_id; we need the reverse
    for (auto& [token, id] : vocab_json.items()) {
        id_to_token[id.get<int>()] = token;
    }

    std::cout << "Loaded vocabulary with " << id_to_token.size() << " tokens\n";
    return id_to_token;
}

// Normalize audio: zero mean, unit variance (standard for CTC models)
std::vector<float> normalize_audio(const std::vector<float>& audio) {
    if (audio.empty()) return audio;

    double sum = std::accumulate(audio.begin(), audio.end(), 0.0);
    double mean = sum / audio.size();

    double sq_sum = 0.0;
    for (float s : audio) {
        sq_sum += (s - mean) * (s - mean);
    }
    double std_dev = std::sqrt(sq_sum / audio.size());
    if (std_dev < 1e-7) std_dev = 1e-7;  // avoid division by zero

    std::vector<float> normalized(audio.size());
    for (size_t i = 0; i < audio.size(); i++) {
        normalized[i] = static_cast<float>((audio[i] - mean) / std_dev);
    }
    return normalized;
}

// CTC greedy decoding: argmax -> collapse repeats -> remove blank tokens
std::string ctc_greedy_decode(const float* logits, size_t time_steps, size_t vocab_size,
                               const std::unordered_map<int, std::string>& vocab,
                               int blank_id = 0) {
    std::string result;
    int prev_token = -1;

    for (size_t t = 0; t < time_steps; t++) {
        // Find argmax for this time step
        int best_id = 0;
        float best_score = logits[t * vocab_size];
        for (size_t v = 1; v < vocab_size; v++) {
            if (logits[t * vocab_size + v] > best_score) {
                best_score = logits[t * vocab_size + v];
                best_id = static_cast<int>(v);
            }
        }

        // CTC collapse: skip blanks and repeated tokens
        if (best_id != blank_id && best_id != prev_token) {
            auto it = vocab.find(best_id);
            if (it != vocab.end()) {
                std::string token = it->second;
                // Handle special word-piece tokens (e.g., "▁" prefix = space)
                if (!token.empty() && token[0] == '|') {
                    // Pipe character often used as word boundary in CTC vocabs
                    result += " ";
                } else {
                    result += token;
                }
            }
        }
        prev_token = best_id;
    }

    // Trim leading/trailing whitespace
    size_t start = result.find_first_not_of(' ');
    size_t end = result.find_last_not_of(' ');
    if (start == std::string::npos) return "";
    return result.substr(start, end - start + 1);
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

    // Validate paths
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

    // Read and normalize audio
    std::cout << "Reading audio: " << wav_path << "\n";
    auto raw_audio = utils::audio::read_wav(wav_path);
    std::cout << "Audio duration: " << std::fixed << std::setprecision(1)
              << (raw_audio.size() / 16000.0) << " seconds ("
              << raw_audio.size() << " samples)\n";

    auto audio = normalize_audio(raw_audio);

    // Initialize OpenVINO Runtime
    std::cout << "Device: " << device << "\n\n";
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

    // Configure device-specific options
    ov::AnyMap config;
    if (device == "GPU" || device.find("GPU") != std::string::npos) {
        config.insert({ov::cache_dir("medasr_cache")});
    }

    auto compiled_model = core.compile_model(model, device, config);
    auto infer_request = compiled_model.create_infer_request();

    // Prepare input tensor
    auto input = compiled_model.input(0);
    std::cout << "Input shape: " << input.get_shape() << "\n";

    // Create input tensor [1, audio_length] or [1, 1, audio_length]
    auto input_shape = input.get_partial_shape();
    ov::Shape tensor_shape;
    if (input_shape.size() == 2) {
        tensor_shape = {1, audio.size()};
    } else if (input_shape.size() == 3) {
        tensor_shape = {1, 1, audio.size()};
    } else {
        tensor_shape = {1, audio.size()};
    }

    ov::Tensor input_tensor(ov::element::f32, tensor_shape, audio.data());
    infer_request.set_input_tensor(input_tensor);

    // Run inference
    std::cout << "Running inference...\n";
    auto start_time = std::chrono::high_resolution_clock::now();
    infer_request.infer();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Get output and decode
    auto output_tensor = infer_request.get_output_tensor();
    auto output_shape = output_tensor.get_shape();
    std::cout << "Output shape: " << output_shape << "\n";

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
              << (raw_audio.size() / 16000.0) << " s\n";
    std::cout << "Real-time factor: " << std::fixed << std::setprecision(2)
              << (duration.count() / 1000.0) / (raw_audio.size() / 16000.0) << "x\n";

    std::cout << "\nRemember: Always verify medical transcriptions with qualified professionals.\n";

    return EXIT_SUCCESS;

} catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Unknown error occurred\n";
    return EXIT_FAILURE;
}
