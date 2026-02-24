# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Standalone CMakeLists.txt for MedGemma Medical Image Analysis Sample
# This builds the medgemma_medical_chat sample independently,
# without requiring the full openvino.genai source tree.
#
# Prerequisites:
#   - OpenVINO GenAI installed (provides openvino::genai CMake target)
#   - CMake 3.16+
#   - C++17 compliant compiler
#
# Build:
#   cmake -B build
#   cmake --build build --config Release

cmake_minimum_required(VERSION 3.16)
project(medgemma_medical_chat_sample LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# Find the OpenVINO GenAI package
find_package(OpenVINOGenAI REQUIRED
    PATHS
        ${OpenVINO_DIR}
    NO_CMAKE_FIND_ROOT_PATH
)

# Download stb_image.h (single-header image loading library)
file(DOWNLOAD
    https://raw.githubusercontent.com/nothings/stb/f75e8d1cad7d90d72ef7a4661f1b994ef78b4e31/stb_image.h
    ${CMAKE_BINARY_DIR}/stb_image.h
    EXPECTED_HASH MD5=27932e6fb3a2f26aee2fc33f2cb4e696)

# MedGemma Medical Chat Sample
add_executable(medgemma_medical_chat
    medgemma_medical_chat.cpp
    load_image.cpp
)

target_include_directories(medgemma_medical_chat PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_BINARY_DIR}"
)

target_link_libraries(medgemma_medical_chat PRIVATE openvino::genai)

set_target_properties(medgemma_medical_chat PROPERTIES
    COMPILE_PDB_NAME medgemma_medical_chat
    FOLDER samples
    INSTALL_RPATH_USE_LINK_PATH ON
)

install(TARGETS medgemma_medical_chat
    RUNTIME DESTINATION bin/
    COMPONENT samples_bin
)

message(STATUS "MedGemma Medical Chat sample configured successfully")
message(STATUS "  Build with: cmake --build ${CMAKE_BINARY_DIR} --config Release")
