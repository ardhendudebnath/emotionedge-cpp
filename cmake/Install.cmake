# `cmake --install <build> [--prefix DIR]`: the apps, the data they read, and the licences that
# have to go with them. The layout:
#   bin/emotionedge, bin/emotionedge-desktop
#   lib/                        whisper.cpp's shared libraries, when it is built
#   share/emotionedge/config/   the pipeline configs (paths in them are relative to the file)
#   share/emotionedge/models/   manifest.json and fetch_models.py; models are fetched next to them
#   share/emotionedge/fonts/    the desktop app's caption font, with its OFL
#   share/doc/EmotionEdge/      THIRD_PARTY_NOTICES.md, licenses.md and the licence texts
# The apps find share/emotionedge from their own location (core/runtime/install_paths.hpp).
# Fetched sources install nothing of their own (ee_make_available, cmake/Dependencies.cmake).
if(NOT EE_BUILD_APPS)
    return()
endif()

install(TARGETS emotionedge RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(DIRECTORY "${PROJECT_SOURCE_DIR}/config/" DESTINATION "${EE_INSTALL_DATADIR}/config")
install(FILES
    "${PROJECT_SOURCE_DIR}/models/manifest.json"
    "${PROJECT_SOURCE_DIR}/models/README.md"
    DESTINATION "${EE_INSTALL_DATADIR}/models")
install(PROGRAMS "${PROJECT_SOURCE_DIR}/models/fetch_models.py" DESTINATION "${EE_INSTALL_DATADIR}/models")
if(EE_WITH_DESKTOP)
    install(TARGETS emotionedge-desktop RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
    install(FILES "${EE_DESKTOP_FONT}" "${CMAKE_BINARY_DIR}/fonts/OFL.txt" DESTINATION "${EE_INSTALL_DATADIR}/fonts")
endif()

# The shared libraries built here: whisper.cpp builds itself and ggml shared (cmake/Backends.cmake).
# They go to lib/ (bin/ on Windows), where the apps' RPATH, $ORIGIN/../lib, finds them.
set(_shared_libs "")
foreach(_t whisper ggml ggml-base ggml-cpu ggml-cuda ggml-blas)
    if(TARGET ${_t})
        get_target_property(_type ${_t} TYPE)
        if(_type STREQUAL "SHARED_LIBRARY")
            list(APPEND _shared_libs ${_t})
        endif()
    endif()
endforeach()
if(_shared_libs)
    install(TARGETS ${_shared_libs}
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" NAMELINK_SKIP
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
endif()

# ---- Third-party notices ------------------------------------------------------------------------
# One row per component in this build. "compiled in" components are part of the installed
# binaries, so their licence texts are installed with them. "linked" ones are shared libraries
# the binaries load from where they were found. They are not installed, and whoever ships them
# alongside must add their notices.
set(_notices "")

# ee_notice(<name> <licence> <compiled in|linked> [FILES <file>...] [NOTE <text>])
# Installs each existing FILE under share/doc/EmotionEdge/licenses/<name>/.
function(ee_notice name license how)
    cmake_parse_arguments(N "" "NOTE" "FILES" ${ARGN})
    set(texts "")
    foreach(f IN LISTS N_FILES)
        if(f AND EXISTS "${f}" AND NOT IS_DIRECTORY "${f}")
            get_filename_component(base "${f}" NAME)
            install(FILES "${f}" DESTINATION "${CMAKE_INSTALL_DOCDIR}/licenses/${name}")
            list(APPEND texts "`licenses/${name}/${base}`")
        endif()
    endforeach()
    if(texts)
        list(JOIN texts ", " texts)
    else()
        set(texts "not found in this build: see the project's site")
    endif()
    if(N_NOTE)
        string(APPEND texts ". ${N_NOTE}")
    endif()
    set(_notices "${_notices}| ${name} | ${license} | ${how} | ${texts} |\n" PARENT_SCOPE)
endfunction()

# A Debian package's copyright file, for a library installed by apt (or unpacked from its .deb):
# <prefix>/lib[/<multiarch>]/libX.so -> <prefix>/share/doc/<package>/copyright.
function(ee_debian_copyright out library)
    set(hints "")
    if(library)
        get_filename_component(dir "${library}" DIRECTORY)
        foreach(pkg IN LISTS ARGN)
            list(APPEND hints "${dir}/../share/doc/${pkg}" "${dir}/../../share/doc/${pkg}")
        endforeach()
    endif()
    find_file(_copyright NAMES copyright PATHS ${hints} NO_DEFAULT_PATH NO_CACHE)
    set(${out} "${_copyright}" PARENT_SCOPE)
endfunction()

# Always: the config parser and the JSON library (fetched, or the system's).
ee_notice(yaml-cpp MIT "compiled in" FILES "${yaml-cpp_SOURCE_DIR}/LICENSE")
ee_notice(nlohmann-json MIT "compiled in" FILES "${nlohmann_json_SOURCE_DIR}/LICENSE.MIT")

if(EE_WITH_ONNXRUNTIME)
    ee_notice(onnxruntime MIT linked
        FILES "${EE_ONNXRUNTIME_ROOT}/LICENSE" "${EE_ONNXRUNTIME_ROOT}/ThirdPartyNotices.txt")
endif()
if(EE_WITH_WHISPER)
    if("whisper" IN_LIST _shared_libs)
        set(_how "installed (shared libraries in lib/)")
    else()
        set(_how "compiled in")
    endif()
    ee_notice(whisper.cpp MIT "${_how}" FILES "${whisper_SOURCE_DIR}/LICENSE" NOTE "Includes ggml (MIT).")
    if(EE_WHISPER_CUDA)
        ee_notice(cuda "NVIDIA CUDA Toolkit EULA" linked NOTE "The CUDA runtime and cuBLAS, used by ggml's CUDA backend.")
    endif()
endif()
if(EE_WITH_CTRANSLATE2)
    ee_notice(ctranslate2 MIT linked NOTE "Its own math libraries (oneDNN, OpenBLAS or MKL) come with it.")
    if(DEFINED sentencepiece_SOURCE_DIR)
        ee_notice(sentencepiece Apache-2.0 "compiled in" FILES "${sentencepiece_SOURCE_DIR}/LICENSE")
    else()
        ee_debian_copyright(_spm "${EE_SPM_LIBRARY}" libsentencepiece0 libsentencepiece-dev)
        ee_notice(sentencepiece Apache-2.0 linked FILES "${_spm}")
    endif()
endif()
if(EE_WITH_PIPER)
    ee_debian_copyright(_espeak "${EE_ESPEAK_LIBRARY}" libespeak-ng1 libespeak-ng-dev)
    get_filename_component(_espeak_dir "${EE_ESPEAK_LIBRARY}" DIRECTORY)
    find_file(_gpl3 NAMES GPL-3
        PATHS "${_espeak_dir}/../share/common-licenses" "${_espeak_dir}/../../share/common-licenses" /usr/share/common-licenses
        NO_DEFAULT_PATH NO_CACHE)
    ee_notice(espeak-ng GPL-3.0-or-later linked FILES "${_espeak}" "${_gpl3}"
        NOTE "**Binaries that link it fall under the GPL-3.0 when distributed.**")
    if(NOT _gpl3)
        message(WARNING "The GPL-3.0 text was not found: add it to the notices before distributing this build")
    endif()
endif()
if(EE_WITH_MINIAUDIO)
    ee_notice(miniaudio "Unlicense or MIT-0" "compiled in" FILES "${miniaudio_SOURCE_DIR}/LICENSE"
        NOTE "It loads the system's audio libraries (PulseAudio, ALSA) when it runs.")
endif()
if(EE_WITH_WEBSOCKET)
    ee_notice(ixwebsocket BSD-3-Clause "compiled in" FILES "${ixwebsocket_SOURCE_DIR}/LICENSE.txt")
    if(USE_TLS)
        ee_notice(openssl Apache-2.0 linked NOTE "The system's, for wss://.")
    endif()
endif()
if(EE_WITH_GRPC)
    ee_notice(grpc Apache-2.0 linked NOTE "The system's, with protobuf (BSD-3-Clause), abseil (Apache-2.0), re2 (BSD-3-Clause) and c-ares (MIT).")
endif()
if(EE_WITH_DESKTOP)
    ee_notice(dear-imgui MIT "compiled in" FILES "${imgui_SOURCE_DIR}/LICENSE.txt" NOTE "Includes stb_truetype (MIT or public domain).")
    ee_notice(glfw Zlib "compiled in" FILES "${glfw_SOURCE_DIR}/LICENSE.md")
    ee_notice(harfbuzz "MIT (Old MIT)" "compiled in" FILES "${harfbuzz_SOURCE_DIR}/COPYING")
    ee_notice(noto-sans-devanagari OFL-1.1 "installed (font)" FILES "${CMAKE_BINARY_DIR}/fonts/OFL.txt")
endif()

set(_gpl_note "")
if(EE_WITH_PIPER)
    set(_gpl_note "\nThis build links espeak-ng, which is GPL-3.0-or-later. Distributing it means distributing it under the GPL-3.0, with its corresponding source.\n")
endif()
file(WRITE "${CMAKE_BINARY_DIR}/THIRD_PARTY_NOTICES.md"
"# Third-party notices: EmotionEdge ${PROJECT_VERSION}

These components are in this build of EmotionEdge.
- **compiled in:** part of the installed binaries, so their licence texts are installed here, under `licenses/`.
- **installed:** files of their own in this install (shared libraries, the font), with their licence texts under `licenses/`.
- **linked:** shared libraries the binaries load from where they were found when they were built. They are not installed. Whoever ships them alongside must add their notices.
${_gpl_note}
| Component | Licence | In this install | Licence text |
|---|---|---|---|
${_notices}
EmotionEdge itself has no licence yet. Models are not part of an install either. `share/emotionedge/models/manifest.json` lists each model's licence, and `licenses.md` says what each build may be used for.
")
install(FILES "${CMAKE_BINARY_DIR}/THIRD_PARTY_NOTICES.md" "${PROJECT_SOURCE_DIR}/docs/licenses.md"
    DESTINATION "${CMAKE_INSTALL_DOCDIR}")
