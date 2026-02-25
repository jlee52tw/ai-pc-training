// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

namespace utils {
namespace audio {

// Read a 16kHz mono WAV file and return normalized float samples
std::vector<float> read_wav(const std::string& filename);

}  // namespace audio
}  // namespace utils
