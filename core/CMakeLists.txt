# One static library per blueprint module. Include paths are rooted at the repository, so code
# includes "core/<module>/<header>.hpp".
function(ee_add_module name)
    cmake_parse_arguments(ARG "" "" "SOURCES;PUBLIC_DEPS;PRIVATE_DEPS" ${ARGN})
    add_library(ee_${name} STATIC ${ARG_SOURCES})
    add_library(ee::${name} ALIAS ee_${name})
    target_include_directories(ee_${name} PUBLIC "${PROJECT_SOURCE_DIR}")
    target_link_libraries(ee_${name}
        PUBLIC ee::project_options ${ARG_PUBLIC_DEPS}
        PRIVATE ${ARG_PRIVATE_DEPS})
    set_target_properties(ee_${name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endfunction()

# ---- telemetry/: metrics, latency budget, ECS, Perfetto trace ---------------------------------
ee_add_module(telemetry
    SOURCES
        telemetry/metrics.cpp
        telemetry/telemetry.cpp
        telemetry/trace.cpp
    PUBLIC_DEPS Threads::Threads)

# ---- runtime/: frames, queues, scheduler, config, model registry, inference runtime -----------
set(EE_RUNTIME_SOURCES
    runtime/config.cpp
    runtime/frame.cpp
    runtime/graph.cpp
    runtime/log.cpp
    runtime/model_registry.cpp
    runtime/params.cpp
    runtime/sha256.cpp
    runtime/stage_registry.cpp
    runtime/thread_util.cpp)
set(EE_RUNTIME_DEPS ee::telemetry Threads::Threads)
if(EE_WITH_ONNXRUNTIME)
    list(APPEND EE_RUNTIME_SOURCES runtime/onnx.cpp)
    list(APPEND EE_RUNTIME_DEPS ee::onnxruntime)
endif()
ee_add_module(runtime
    SOURCES ${EE_RUNTIME_SOURCES}
    PUBLIC_DEPS ${EE_RUNTIME_DEPS}
    PRIVATE_DEPS yaml-cpp::yaml-cpp nlohmann_json::nlohmann_json)

# ---- audio/: capture, ring, DSP, VAD & segmenter, playback, speaker encoder -------------------
set(EE_AUDIO_SOURCES
    audio/audio_io.cpp
    audio/dsp.cpp
    audio/frontend.cpp
    audio/pitch.cpp
    audio/playback.cpp
    audio/resampler.cpp
    audio/segmenter.cpp
    audio/speaker.cpp
    audio/speech_detector.cpp
    audio/wav.cpp)
set(EE_AUDIO_DEPS ee::runtime)
if(EE_WITH_ONNXRUNTIME)
    list(APPEND EE_AUDIO_SOURCES audio/silero_detector.cpp)
endif()
if(EE_WITH_MINIAUDIO)
    list(APPEND EE_AUDIO_SOURCES audio/device.cpp)
    list(APPEND EE_AUDIO_DEPS ee::miniaudio)
endif()
ee_add_module(audio SOURCES ${EE_AUDIO_SOURCES} PUBLIC_DEPS ${EE_AUDIO_DEPS})

# ---- emotion/: acoustic, lexical, fusion, state tracker (3.1), consistency (5.2) --------------
set(EE_EMOTION_SOURCES
    emotion/acoustic.cpp
    emotion/consistency_stage.cpp
    emotion/emotion_stage.cpp
    emotion/fusion.cpp
    emotion/lexical.cpp
    emotion/prosody_features.cpp
    emotion/state_tracker.cpp)
if(EE_WITH_ONNXRUNTIME)
    list(APPEND EE_EMOTION_SOURCES emotion/onnx_acoustic.cpp)
endif()
ee_add_module(emotion SOURCES ${EE_EMOTION_SOURCES} PUBLIC_DEPS ee::audio ee::runtime)

# ---- asr/: streaming ASR, LocalAgreement-2, whisper.cpp adapter ------------------------------
set(EE_ASR_SOURCES
    asr/asr_stage.cpp
    asr/local_agreement.cpp
    asr/scripted_engine.cpp)
set(EE_ASR_DEPS ee::runtime)
if(EE_WITH_WHISPER)
    list(APPEND EE_ASR_SOURCES asr/whisper_engine.cpp)
    list(APPEND EE_ASR_DEPS ee::whisper)
endif()
ee_add_module(asr SOURCES ${EE_ASR_SOURCES} PUBLIC_DEPS ${EE_ASR_DEPS} PRIVATE_DEPS nlohmann_json::nlohmann_json)

# ---- prosody/: expressivity profiles (3.3), emotion controller (4.1), style vectors ------------
ee_add_module(prosody
    SOURCES
        prosody/controller.cpp
        prosody/controller_stage.cpp
        prosody/expressivity.cpp
        prosody/style.cpp
    PUBLIC_DEPS ee::runtime
    PRIVATE_DEPS yaml-cpp::yaml-cpp nlohmann_json::nlohmann_json)

# ---- translate/: CT2 + control tokens, glossary, wait-k, emphasis projection (3.2) ------------
set(EE_TRANSLATE_SOURCES
    translate/control_tokens.cpp
    translate/emphasis.cpp
    translate/glossary.cpp
    translate/languages.cpp
    translate/phrasebook.cpp
    translate/translate_stage.cpp
    translate/wait_k.cpp)
set(EE_TRANSLATE_DEPS ee::prosody ee::asr ee::runtime)
if(EE_WITH_CTRANSLATE2)
    list(APPEND EE_TRANSLATE_SOURCES translate/ct2_translator.cpp)
    list(APPEND EE_TRANSLATE_DEPS ee::ctranslate2)
endif()
ee_add_module(translate SOURCES ${EE_TRANSLATE_SOURCES} PUBLIC_DEPS ${EE_TRANSLATE_DEPS} PRIVATE_DEPS yaml-cpp::yaml-cpp)

# ---- tts/: StyleTTS2 / Piper slot, formant stand-in, clause chunker (4.2) ---------------------
set(EE_TTS_SOURCES
    tts/clause_chunker.cpp
    tts/formant_synth.cpp
    tts/tts_stage.cpp)
set(EE_TTS_DEPS ee::translate ee::audio ee::prosody ee::runtime)
if(EE_WITH_PIPER)
    list(APPEND EE_TTS_SOURCES tts/piper_engine.cpp)
    list(APPEND EE_TTS_DEPS ee::piper)
endif()
ee_add_module(tts SOURCES ${EE_TTS_SOURCES} PUBLIC_DEPS ${EE_TTS_DEPS} PRIVATE_DEPS nlohmann_json::nlohmann_json)

# ---- pipeline/: built-in stage registry, outputs (5.3), session runner, demo input ------------
ee_add_module(pipeline
    SOURCES
        pipeline/builtin_stages.cpp
        pipeline/demo.cpp
        pipeline/recorder.cpp
        pipeline/session.cpp
    PUBLIC_DEPS ee::tts ee::translate ee::prosody ee::asr ee::emotion ee::audio ee::runtime ee::telemetry
    PRIVATE_DEPS nlohmann_json::nlohmann_json)
