# apps/: the emotionedge CLI. The desktop UI (Dear ImGui) and the gRPC/WebSocket server are
# roadmap phase 5; see apps/README.md.
add_executable(emotionedge cli/main.cpp)
target_link_libraries(emotionedge PRIVATE ee::pipeline ee_project_options nlohmann_json::nlohmann_json)
target_compile_definitions(emotionedge PRIVATE
    EE_DEFAULT_CONFIG="${PROJECT_SOURCE_DIR}/config/pipeline.yaml"
    EE_VERSION="${PROJECT_VERSION}")
