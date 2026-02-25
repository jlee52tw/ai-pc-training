# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Standalone CMakeLists.txt for MedASR Medical Speech Recognition Sample
#
# This builds the medasr_medical_asr sample using OpenVINO Runtime
# for inference on a Conformer-CTC speech recognition model.
#
# Prerequisites:
#   - OpenVINO GenAI installed (provides OpenVINO Runtime via openvino::genai)
#   - CMake 3.16+
#   - C++17 compliant compiler
#
# Build:
#   cmake -B build
#   cmake --build build --config Release

cmake_minimum_required(VERSION 3.16)
project(medasr_medical_asr_sample LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# Find the OpenVINO GenAI package (provides OpenVINO Runtime)
find_package(OpenVINOGenAI REQUIRED
    PATHS
        ${OpenVINO_DIR}
    NO_CMAKE_FIND_ROOT_PATH
)

# Fetch dr_wav (single-header WAV reading library)
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()
if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
endif()

include(FetchContent)

if(NOT TARGET dr_libs)
    FetchContent_Declare(dr_libs
        URL https://github.com/mackron/dr_libs/archive/da35f9d6c7374a95353fd1df1d394d44ab66cf01.tar.gz
        URL_HASH SHA256=2704d347f480ca1bc92233fb01747e4550cc8031735b6ea62ca9990ebb8851ae)
    FetchContent_MakeAvailable(dr_libs)
endif()

# Fetch nlohmann/json for vocabulary loading
FetchContent_Declare(json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d)
FetchContent_MakeAvailable(json)

# MedASR Medical Speech Recognition Sample
add_executable(medasr_medical_asr
    medasr_medical_asr.cpp
    audio_utils.cpp
)

target_include_directories(medasr_medical_asr PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "$<BUILD_INTERFACE:${dr_libs_SOURCE_DIR}>"
)

target_link_libraries(medasr_medical_asr PRIVATE
    openvino::genai
    nlohmann_json::nlohmann_json
)

set_target_properties(medasr_medical_asr PROPERTIES
    COMPILE_PDB_NAME medasr_medical_asr
    FOLDER samples
    INSTALL_RPATH_USE_LINK_PATH ON
)

install(TARGETS medasr_medical_asr
    RUNTIME DESTINATION bin/
    COMPONENT samples_bin
)

message(STATUS "MedASR Medical ASR sample configured successfully")
message(STATUS "  Build with: cmake --build ${CMAKE_BINARY_DIR} --config Release")
