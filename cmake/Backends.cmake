# Optional inference and audio backends. Each enabled backend defines an `ee::<name>` interface
# target (with an EE_HAVE_<NAME>=1 definition) that the matching engine adapter links against.

if(EE_WITH_PIPER AND NOT EE_WITH_ONNXRUNTIME)
    message(STATUS "EE_WITH_PIPER needs ONNX Runtime; enabling EE_WITH_ONNXRUNTIME")
    set(EE_WITH_ONNXRUNTIME ON CACHE BOOL "" FORCE)
endif()

# ---- ONNX Runtime (Silero VAD, emotion2vec head, Piper voices) ---------------------------------
if(EE_WITH_ONNXRUNTIME)
    set(EE_ONNXRUNTIME_ROOT "" CACHE PATH "Extracted ONNX Runtime release containing include/ and lib/")
    find_package(onnxruntime CONFIG QUIET)
    add_library(ee_onnxruntime INTERFACE)
    if(TARGET onnxruntime::onnxruntime)
        target_link_libraries(ee_onnxruntime INTERFACE onnxruntime::onnxruntime)
    else()
        find_path(EE_ORT_INCLUDE_DIR onnxruntime_cxx_api.h
            HINTS "${EE_ONNXRUNTIME_ROOT}/include"
            PATH_SUFFIXES onnxruntime onnxruntime/core/session)
        find_library(EE_ORT_LIBRARY NAMES onnxruntime HINTS "${EE_ONNXRUNTIME_ROOT}/lib")
        if(NOT EE_ORT_INCLUDE_DIR OR NOT EE_ORT_LIBRARY)
            message(FATAL_ERROR
                "ONNX Runtime not found. Extract a release from "
                "https://github.com/microsoft/onnxruntime/releases and configure with "
                "-DEE_ONNXRUNTIME_ROOT=<dir>, or use the vcpkg 'onnxruntime' feature.")
        endif()
        target_include_directories(ee_onnxruntime SYSTEM INTERFACE "${EE_ORT_INCLUDE_DIR}")
        target_link_libraries(ee_onnxruntime INTERFACE "${EE_ORT_LIBRARY}")
    endif()
    target_compile_definitions(ee_onnxruntime INTERFACE EE_HAVE_ONNXRUNTIME=1)
    add_library(ee::onnxruntime ALIAS ee_onnxruntime)
endif()

# ---- whisper.cpp (streaming ASR) ----------------------------------------------------------------
if(EE_WITH_WHISPER)
    find_package(whisper CONFIG QUIET)
    if(NOT TARGET whisper)
        if(NOT EE_FETCH_DEPS)
            message(FATAL_ERROR "whisper.cpp not found and EE_FETCH_DEPS=OFF")
        endif()
        set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)
        # Keep whisper/ggml static so their option(BUILD_SHARED_LIBS) does not leak into our targets.
        set(BUILD_SHARED_LIBS OFF)
        FetchContent_Declare(whisper
            # v1.9.4
            URL https://github.com/ggml-org/whisper.cpp/archive/927cfce34f31707e17f2bff35c349632fb9e2c3a.tar.gz)
        FetchContent_MakeAvailable(whisper)
    endif()
    add_library(ee_whisper INTERFACE)
    target_link_libraries(ee_whisper INTERFACE whisper)
    target_compile_definitions(ee_whisper INTERFACE EE_HAVE_WHISPER=1)
    add_library(ee::whisper ALIAS ee_whisper)
endif()

# ---- CTranslate2 + SentencePiece (NLLB-200 translation) ----------------------------------------
if(EE_WITH_CTRANSLATE2)
    find_package(ctranslate2 CONFIG REQUIRED)
    find_path(EE_SPM_INCLUDE_DIR sentencepiece_processor.h)
    find_library(EE_SPM_LIBRARY NAMES sentencepiece)
    if(NOT EE_SPM_INCLUDE_DIR OR NOT EE_SPM_LIBRARY)
        message(FATAL_ERROR "SentencePiece not found (apt: libsentencepiece-dev, vcpkg: sentencepiece)")
    endif()
    add_library(ee_ctranslate2 INTERFACE)
    target_include_directories(ee_ctranslate2 SYSTEM INTERFACE "${EE_SPM_INCLUDE_DIR}")
    target_link_libraries(ee_ctranslate2 INTERFACE CTranslate2::ctranslate2 "${EE_SPM_LIBRARY}")
    target_compile_definitions(ee_ctranslate2 INTERFACE EE_HAVE_CTRANSLATE2=1)
    add_library(ee::ctranslate2 ALIAS ee_ctranslate2)
endif()

# ---- espeak-ng (phonemizer for Piper voices) ----------------------------------------------------
if(EE_WITH_PIPER)
    find_path(EE_ESPEAK_INCLUDE_DIR espeak-ng/speak_lib.h)
    find_library(EE_ESPEAK_LIBRARY NAMES espeak-ng)
    if(NOT EE_ESPEAK_INCLUDE_DIR OR NOT EE_ESPEAK_LIBRARY)
        message(FATAL_ERROR "espeak-ng not found (apt: libespeak-ng-dev)")
    endif()
    add_library(ee_piper INTERFACE)
    target_include_directories(ee_piper SYSTEM INTERFACE "${EE_ESPEAK_INCLUDE_DIR}")
    target_link_libraries(ee_piper INTERFACE ee::onnxruntime "${EE_ESPEAK_LIBRARY}")
    target_compile_definitions(ee_piper INTERFACE EE_HAVE_PIPER=1)
    add_library(ee::piper ALIAS ee_piper)
endif()

# ---- miniaudio (live capture / playback, T0) ----------------------------------------------------
if(EE_WITH_MINIAUDIO)
    find_path(EE_MINIAUDIO_INCLUDE_DIR miniaudio.h PATH_SUFFIXES miniaudio)
    if(NOT EE_MINIAUDIO_INCLUDE_DIR)
        if(NOT EE_FETCH_DEPS)
            message(FATAL_ERROR "miniaudio.h not found and EE_FETCH_DEPS=OFF")
        endif()
        # SOURCE_SUBDIR points nowhere so only the header is used; miniaudio's own CMake is skipped.
        FetchContent_Declare(miniaudio
            # 0.11.25
            URL https://github.com/mackron/miniaudio/archive/9634bedb5b5a2ca38c1ee7108a9358a4e233f14d.tar.gz
            SOURCE_SUBDIR header-only)
        FetchContent_MakeAvailable(miniaudio)
        set(EE_MINIAUDIO_INCLUDE_DIR "${miniaudio_SOURCE_DIR}")
    endif()
    add_library(ee_miniaudio INTERFACE)
    target_include_directories(ee_miniaudio SYSTEM INTERFACE "${EE_MINIAUDIO_INCLUDE_DIR}")
    target_link_libraries(ee_miniaudio INTERFACE Threads::Threads ${CMAKE_DL_LIBS})
    target_compile_definitions(ee_miniaudio INTERFACE EE_HAVE_MINIAUDIO=1)
    add_library(ee::miniaudio ALIAS ee_miniaudio)
endif()
