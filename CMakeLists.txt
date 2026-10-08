cmake_minimum_required(VERSION 3.24)

project(EmotionEdge
    VERSION 0.1.0
    DESCRIPTION "Real-time, emotion-preserving speech translation on the edge"
    HOMEPAGE_URL "https://github.com/ardhendudebnath/emotionedge-cpp"
    LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---- Options ------------------------------------------------------------------------------------
option(EE_BUILD_TESTS "Build GoogleTest unit and golden-audio tests" ${PROJECT_IS_TOP_LEVEL})
option(EE_BUILD_APPS "Build apps/ (emotionedge CLI)" ON)
option(EE_BUILD_BENCH "Build bench/ latency harness" ON)
option(EE_FETCH_DEPS "Fetch missing third-party sources with FetchContent" ON)
option(EE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
set(EE_SANITIZE "" CACHE STRING "Sanitizers to enable, e.g. 'address,undefined' or 'thread'")

# Inference and audio backends. All OFF by default: the core, the stub engines and the
# tests build with no native ML dependencies. Each backend adds a real engine adapter.
option(EE_WITH_ONNXRUNTIME "ONNX Runtime engines: Silero VAD, emotion2vec head" OFF)
option(EE_WITH_WHISPER "whisper.cpp streaming ASR adapter" OFF)
option(EE_WHISPER_CUDA "Build whisper.cpp with its CUDA backend (needs the CUDA toolkit; with EE_WITH_WHISPER)" OFF)
option(EE_WITH_CTRANSLATE2 "CTranslate2 + SentencePiece translation adapter (NLLB-200)" OFF)
option(EE_WITH_PIPER "Piper TTS adapter (ONNX Runtime + espeak-ng)" OFF)
option(EE_WITH_MINIAUDIO "miniaudio live capture and playback" OFF)
option(EE_WITH_WEBSOCKET "IXWebSocket streaming server (emotionedge serve)" OFF)
option(EE_WITH_GRPC "gRPC streaming server (emotionedge serve --grpc-port): needs gRPC and protobuf installed" OFF)
option(EE_WITH_DESKTOP "Desktop app emotionedge-desktop: Dear ImGui + GLFW, HarfBuzz text shaping (needs EE_WITH_MINIAUDIO)" OFF)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(Warnings)
include(Dependencies)
include(Backends)

if(EE_SANITIZE)
    if(MSVC)
        message(WARNING "EE_SANITIZE is only supported with GCC/Clang; ignoring")
    else()
        add_compile_options(-fsanitize=${EE_SANITIZE} -fno-omit-frame-pointer)
        add_link_options(-fsanitize=${EE_SANITIZE})
    endif()
endif()

add_subdirectory(core)

if(EE_BUILD_APPS)
    add_subdirectory(apps)
endif()

if(EE_BUILD_BENCH)
    add_subdirectory(bench)
endif()

if(EE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

message(STATUS "EmotionEdge ${PROJECT_VERSION}: onnxruntime=${EE_WITH_ONNXRUNTIME} whisper=${EE_WITH_WHISPER} "
               "ctranslate2=${EE_WITH_CTRANSLATE2} piper=${EE_WITH_PIPER} miniaudio=${EE_WITH_MINIAUDIO} "
               "websocket=${EE_WITH_WEBSOCKET} grpc=${EE_WITH_GRPC} desktop=${EE_WITH_DESKTOP}")
