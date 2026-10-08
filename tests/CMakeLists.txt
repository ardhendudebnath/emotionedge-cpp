if(EE_FETCH_DEPS)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_Declare(googletest
        URL https://github.com/google/googletest/releases/download/v1.18.0/googletest-1.18.0.tar.gz
        URL_HASH SHA256=6e3191c1455468b3fc35a417fb565c1c5071aee1b7e7f85e30cf48a98d37d8b5
        FIND_PACKAGE_ARGS NAMES GTest)
    FetchContent_MakeAvailable(googletest)
else()
    find_package(GTest CONFIG REQUIRED)
endif()
include(GoogleTest)

# Unit tests per module, plus end-to-end "golden audio" tests of the whole pipeline.
add_executable(ee_tests
    asr/test_asr.cpp
    audio/test_dsp_pitch.cpp
    audio/test_frontend_playback.cpp
    audio/test_ring_wav_resampler.cpp
    audio/test_segmenter.cpp
    emotion/test_classifier_models.cpp
    emotion/test_emotion_models.cpp
    emotion/test_emotion_stages.cpp
    pipeline/test_pipeline_e2e.cpp
    prosody/test_controller.cpp
    server/test_session_pool.cpp
    runtime/test_config.cpp
    runtime/test_graph.cpp
    runtime/test_model_registry.cpp
    runtime/test_sha256.cpp
    runtime/test_spsc_queue.cpp
    runtime/test_telemetry.cpp
    translate/test_translate.cpp
    tts/test_kokoro_g2p.cpp
    tts/test_tts.cpp)

if(EE_WITH_WEBSOCKET)
    target_sources(ee_tests PRIVATE server/test_websocket_server.cpp)
    target_link_libraries(ee_tests PRIVATE ee::websocket)
endif()
if(EE_WITH_GRPC)
    target_sources(ee_tests PRIVATE server/test_grpc_server.cpp)
    target_link_libraries(ee_tests PRIVATE ee::grpc_proto)
endif()

target_include_directories(ee_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(ee_tests PRIVATE ee::pipeline ee::server nlohmann_json::nlohmann_json GTest::gtest_main)
target_compile_definitions(ee_tests PRIVATE EE_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

gtest_discover_tests(ee_tests DISCOVERY_MODE PRE_TEST PROPERTIES TIMEOUT 120)
