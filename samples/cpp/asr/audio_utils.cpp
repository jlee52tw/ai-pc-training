// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "audio_utils.hpp"

#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#define DR_WAV_IMPLEMENTATION
#include <dr_wav.h>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace utils {
namespace audio {

static constexpr unsigned int SAMPLE_RATE = 16000;

std::vector<float> read_wav(const std::string& filename) {
    drwav wav;
    std::vector<uint8_t> wav_data;

    if (filename == "-") {
#ifdef _WIN32
        _setmode(_fileno(stdin), _O_BINARY);
#endif
        uint8_t buf[1024];
        while (true) {
            const size_t n = fread(buf, 1, sizeof(buf), stdin);
            if (n == 0) break;
            wav_data.insert(wav_data.end(), buf, buf + n);
        }
        if (!drwav_init_memory(&wav, wav_data.data(), wav_data.size(), nullptr)) {
            throw std::runtime_error("Failed to open WAV from stdin");
        }
    } else {
        if (!drwav_init_file(&wav, filename.c_str(), nullptr)) {
            throw std::runtime_error("Failed to open WAV file: " + filename);
        }
    }

    if (wav.channels != 1 && wav.channels != 2) {
        drwav_uninit(&wav);
        throw std::runtime_error("WAV file must be mono or stereo");
    }

    if (wav.sampleRate != SAMPLE_RATE) {
        drwav_uninit(&wav);
        throw std::runtime_error("WAV file must be 16kHz (got " +
                                 std::to_string(wav.sampleRate) + " Hz)");
    }

    const uint64_t n = wav.totalPCMFrameCount;
    std::vector<int16_t> pcm16(n * wav.channels);
    drwav_read_pcm_frames_s16(&wav, n, pcm16.data());
    drwav_uninit(&wav);

    std::vector<float> pcmf32(n);
    if (wav.channels == 1) {
        for (uint64_t i = 0; i < n; i++) {
            pcmf32[i] = static_cast<float>(pcm16[i]) / 32768.0f;
        }
    } else {
        for (uint64_t i = 0; i < n; i++) {
            pcmf32[i] = static_cast<float>(pcm16[2 * i] + pcm16[2 * i + 1]) / 65536.0f;
        }
    }

    return pcmf32;
}

std::vector<float> load_mel_filters(const std::string& filename, int rows, int cols) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open mel filter file: " + filename);
    }
    std::vector<float> filters(rows * cols);
    file.read(reinterpret_cast<char*>(filters.data()), filters.size() * sizeof(float));
    if (!file) {
        throw std::runtime_error("Failed to read mel filter data");
    }
    return filters;
}

// Cooley-Tukey radix-2 FFT (in-place, iterative)
static void fft(std::vector<std::complex<double>>& x) {
    size_t N = x.size();
    if (N <= 1) return;

    // Bit-reversal permutation
    for (size_t i = 1, j = 0; i < N; i++) {
        size_t bit = N >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) std::swap(x[i], x[j]);
    }

    // Butterfly operations
    for (size_t len = 2; len <= N; len <<= 1) {
        double angle = -2.0 * M_PI / static_cast<double>(len);
        std::complex<double> wlen(std::cos(angle), std::sin(angle));
        for (size_t i = 0; i < N; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (size_t j = 0; j < len / 2; j++) {
                auto u = x[i + j];
                auto v = x[i + j + len / 2] * w;
                x[i + j] = u + v;
                x[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

// Real FFT: returns n_fft/2 + 1 complex values
static std::vector<std::complex<double>> rfft(const std::vector<double>& input, int n_fft) {
    std::vector<std::complex<double>> x(n_fft);
    for (size_t i = 0; i < static_cast<size_t>(n_fft); i++) {
        x[i] = (i < input.size()) ? std::complex<double>(input[i], 0.0)
                                  : std::complex<double>(0.0, 0.0);
    }
    fft(x);
    // Return only first n_fft/2 + 1 values (positive frequencies)
    int out_size = n_fft / 2 + 1;
    return std::vector<std::complex<double>>(x.begin(), x.begin() + out_size);
}

std::vector<float> compute_mel_spectrogram(const std::vector<float>& audio,
                                            const MelConfig& config,
                                            const std::vector<float>& mel_filters,
                                            size_t& num_frames) {
    int n_fft = config.n_fft;
    int hop = config.hop_length;
    int win_len = config.win_length;
    int n_mels = config.n_mels;
    int freq_bins = n_fft / 2 + 1;  // 257

    // Compute Hann window (periodic=False)
    std::vector<double> window(win_len);
    for (int i = 0; i < win_len; i++) {
        window[i] = 0.5 * (1.0 - std::cos(2.0 * M_PI * i / (win_len - 1)));
    }

    // Compute number of frames (matching torch unfold behavior)
    if (static_cast<int>(audio.size()) < win_len) {
        throw std::runtime_error("Audio too short for mel spectrogram");
    }
    num_frames = (audio.size() - win_len) / hop + 1;

    std::vector<float> mel_spec(num_frames * n_mels);

    for (size_t f = 0; f < num_frames; f++) {
        // Extract windowed frame
        std::vector<double> frame(win_len);
        size_t offset = f * hop;
        for (int i = 0; i < win_len; i++) {
            frame[i] = static_cast<double>(audio[offset + i]) * window[i];
        }

        // FFT (frame is zero-padded to n_fft inside rfft)
        auto spectrum = rfft(frame, n_fft);

        // Power spectrum: |X|^2
        std::vector<double> power(freq_bins);
        for (int i = 0; i < freq_bins; i++) {
            power[i] = std::norm(spectrum[i]);  // |z|^2
        }

        // Apply mel filterbank: power[freq_bins] @ mel_filters[freq_bins, n_mels]
        for (int m = 0; m < n_mels; m++) {
            double val = 0.0;
            for (int k = 0; k < freq_bins; k++) {
                val += power[k] * static_cast<double>(mel_filters[k * n_mels + m]);
            }
            // Clamp and log (matching Python: torch.clamp(mel, min=1e-5) then log)
            val = std::max(val, 1e-5);
            mel_spec[f * n_mels + m] = static_cast<float>(std::log(val));
        }
    }

    return mel_spec;
}

}  // namespace audio
}  // namespace utils
