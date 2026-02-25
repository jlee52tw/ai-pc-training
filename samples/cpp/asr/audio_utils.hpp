// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

namespace utils {
namespace audio {

// Read a 16kHz mono WAV file and return float samples normalized to [-1, 1]
std::vector<float> read_wav(const std::string& filename);

// Mel spectrogram parameters
struct MelConfig {
    int n_fft = 512;
    int hop_length = 160;
    int win_length = 400;
    int n_mels = 128;
    int sample_rate = 16000;
};

// Load mel filterbank matrix from binary file (shape: [n_fft/2+1, n_mels])
std::vector<float> load_mel_filters(const std::string& filename, int rows, int cols);

// Compute log mel spectrogram from audio samples
// Returns a flat vector of shape [num_frames, n_mels] in row-major order
// Also returns number of frames via out parameter
std::vector<float> compute_mel_spectrogram(const std::vector<float>& audio,
                                            const MelConfig& config,
                                            const std::vector<float>& mel_filters,
                                            size_t& num_frames);

}  // namespace audio
}  // namespace utils
