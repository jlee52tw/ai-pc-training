// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "audio_utils.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#define DR_WAV_IMPLEMENTATION
#include <dr_wav.h>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
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

    // Convert to mono float32, normalized to [-1, 1]
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

}  // namespace audio
}  // namespace utils
