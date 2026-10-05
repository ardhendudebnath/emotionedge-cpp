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
        # ggml's CUDA backend. Set CMAKE_CUDA_ARCHITECTURES (e.g. "native", or 120 for Blackwell)
        # to compile for your GPU only; ggml's default list takes much longer.
        set(GGML_CUDA ${EE_WHISPER_CUDA} CACHE BOOL "" FORCE)
        # Keep whisper/ggml static so their option(BUILD_SHARED_LIBS) does not leak into our targets.
        set(BUILD_SHARED_LIBS OFF)
        FetchContent_Declare(whisper
            # v1.9.4
            URL https://github.com/ggml-org/whisper.cpp/archive/927cfce34f31707e17f2bff35c349632fb9e2c3a.tar.gz)
        FetchContent_MakeAvailable(whisper)
        # Third-party headers: keep our warning flags (and -Werror) out of them.
        if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.25)
            foreach(_t whisper ggml ggml-base ggml-cpu ggml-cuda)
                if(TARGET ${_t})
                    set_target_properties(${_t} PROPERTIES SYSTEM ON)
                endif()
            endforeach()
        endif()
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

# ---- IXWebSocket (streaming server, 5.3) --------------------------------------------------------
if(EE_WITH_WEBSOCKET)
    find_package(ixwebsocket CONFIG QUIET)
    if(NOT ixwebsocket_FOUND)
        if(NOT EE_FETCH_DEPS)
            message(FATAL_ERROR "ixwebsocket not found and EE_FETCH_DEPS=OFF")
        endif()
        # Plain ws:// with no compression: no OpenSSL or zlib dependency. Put a TLS proxy in
        # front of the server to expose it beyond localhost.
        set(USE_TLS OFF CACHE BOOL "" FORCE)
        set(USE_ZLIB OFF CACHE BOOL "" FORCE)
        set(IXWEBSOCKET_INSTALL OFF CACHE BOOL "" FORCE)
        set(BUILD_SHARED_LIBS OFF)
        FetchContent_Declare(ixwebsocket
            URL https://github.com/machinezone/IXWebSocket/archive/refs/tags/v12.0.1.tar.gz
            URL_HASH SHA256=d23bdc91dbfe2b9ae13c322d539392d7a6b8b506560f41c90e227fa0f86a2405)
        FetchContent_MakeAvailable(ixwebsocket)
        if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.25)
            set_target_properties(ixwebsocket PROPERTIES SYSTEM ON)
        endif()
        # Its own -Wall -Wextra flag the zlib-less stubs' unused parameters: not ours to fix.
        target_compile_options(ixwebsocket PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-w>)
    endif()
    add_library(ee_websocket INTERFACE)
    target_link_libraries(ee_websocket INTERFACE $<IF:$<TARGET_EXISTS:ixwebsocket::ixwebsocket>,ixwebsocket::ixwebsocket,ixwebsocket>)
    target_compile_definitions(ee_websocket INTERFACE EE_HAVE_WEBSOCKET=1)
    add_library(ee::websocket ALIAS ee_websocket)
endif()

# ---- Desktop app (5.3): Dear ImGui + GLFW, HarfBuzz, Noto Sans Devanagari -------------------------
# Dear ImGui does not shape text, which Hindi needs (vowel signs reorder, consonants join), so
# HarfBuzz shapes the captions. GLFW is built for X11 only: Wayland desktops (and WSLg) run it
# through XWayland. Everything is fetched at pinned, hash-verified versions.
if(EE_WITH_DESKTOP)
    if(NOT EE_WITH_MINIAUDIO)
        message(FATAL_ERROR "EE_WITH_DESKTOP needs EE_WITH_MINIAUDIO (microphone and speaker)")
    endif()
    if(NOT EE_FETCH_DEPS)
        message(FATAL_ERROR "EE_WITH_DESKTOP fetches its dependencies: it needs EE_FETCH_DEPS=ON")
    endif()
    FetchContent_Declare(imgui
        URL https://github.com/ocornut/imgui/archive/refs/tags/v1.92.9b.tar.gz
        URL_HASH SHA256=21d8a0a565e85dce943e375db00812c2f3f0ab21f3f0f7964e364a63422d7f99
        SOURCE_SUBDIR no-cmake)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(glfw
        URL https://github.com/glfw/glfw/releases/download/3.5.1/glfw-3.5.1.zip
        URL_HASH SHA256=ea79bc5feffc254c87291980c2d0bce9acebb68c4983b79f961dcd2cb8a611a0)
    # The shaper only: no FreeType, ICU or GLib, and none of HarfBuzz's extra libraries.
    foreach(_opt HB_BUILD_SUBSET HB_BUILD_RASTER HB_BUILD_VECTOR HB_BUILD_GPU HB_BUILD_UTILS)
        set(${_opt} OFF CACHE BOOL "" FORCE)
    endforeach()
    set(BUILD_SHARED_LIBS OFF)
    FetchContent_Declare(harfbuzz
        URL https://github.com/harfbuzz/harfbuzz/releases/download/14.5.1/harfbuzz-14.5.1.tar.xz
        URL_HASH SHA256=7e2fa4e8c7c98e8d8140671f5772542afaaa6acccfbd746506886b6d85f7f8d6)
    FetchContent_MakeAvailable(imgui glfw harfbuzz)
    foreach(_t glfw harfbuzz)
        if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.25)
            set_target_properties(${_t} PROPERTIES SYSTEM ON)
        endif()
        target_compile_options(${_t} PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-w>)
    endforeach()

    add_library(ee_imgui STATIC
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp)
    target_include_directories(ee_imgui SYSTEM PUBLIC "${imgui_SOURCE_DIR}" "${imgui_SOURCE_DIR}/backends")
    target_link_libraries(ee_imgui PUBLIC glfw ${CMAKE_DL_LIBS})
    # ImGui's OpenGL backend loads GL itself; GLFW must not include the system GL headers.
    target_compile_definitions(ee_imgui PUBLIC GLFW_INCLUDE_NONE)
    target_compile_options(ee_imgui PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-w>)
    add_library(ee::imgui ALIAS ee_imgui)

    # The caption font, under the SIL Open Font License 1.1 (copied next to it).
    set(_font_base https://raw.githubusercontent.com/google/fonts/2bc5d431b584d89e646214bf6cfa75494e76ceea/ofl/notosansdevanagari)
    set(EE_DESKTOP_FONT "${CMAKE_BINARY_DIR}/fonts/NotoSansDevanagari.ttf")
    if(NOT EXISTS "${EE_DESKTOP_FONT}")
        file(DOWNLOAD "${_font_base}/NotoSansDevanagari%5Bwdth%2Cwght%5D.ttf" "${EE_DESKTOP_FONT}"
            EXPECTED_HASH SHA256=14ec4af41f27482216d1c2229f417ff9b1425e1babb014e57d1d40d03229853e TLS_VERIFY ON)
        file(DOWNLOAD "${_font_base}/OFL.txt" "${CMAKE_BINARY_DIR}/fonts/OFL.txt"
            EXPECTED_HASH SHA256=a216f6f8d85c7228093e0ee5e258d9d377e6671f68acb4db1930b29583d0f331 TLS_VERIFY ON)
    endif()
endif()
