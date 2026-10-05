# apps/: the emotionedge CLI, with `serve` (the WebSocket server, 5.3) when built with
# EE_WITH_WEBSOCKET. The desktop UI (Dear ImGui) is roadmap phase 5; see apps/README.md.
add_executable(emotionedge cli/main.cpp)
target_link_libraries(emotionedge PRIVATE ee::pipeline ee_project_options nlohmann_json::nlohmann_json)
if(EE_WITH_WEBSOCKET)
    target_link_libraries(emotionedge PRIVATE ee::server)
    target_compile_definitions(emotionedge PRIVATE EE_HAVE_WEBSOCKET=1)
endif()
target_compile_definitions(emotionedge PRIVATE
    EE_DEFAULT_CONFIG="${PROJECT_SOURCE_DIR}/config/pipeline.yaml"
    EE_VERSION="${PROJECT_VERSION}")
