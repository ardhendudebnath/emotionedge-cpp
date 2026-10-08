# apps/: the emotionedge CLI, with `serve` (the streaming APIs, 5.3) when built with
# EE_WITH_WEBSOCKET and/or EE_WITH_GRPC, and emotionedge-desktop (EE_WITH_DESKTOP). See
# apps/README.md.
add_executable(emotionedge cli/main.cpp)
target_link_libraries(emotionedge PRIVATE ee::pipeline ee::server ee_project_options nlohmann_json::nlohmann_json)
if(EE_WITH_WEBSOCKET)
    target_compile_definitions(emotionedge PRIVATE EE_HAVE_WEBSOCKET=1)
endif()
if(EE_WITH_GRPC)
    target_link_libraries(emotionedge PRIVATE ee::grpc_proto)  # the `stream` client's stubs
endif()
target_compile_definitions(emotionedge PRIVATE
    EE_DEFAULT_CONFIG="${PROJECT_SOURCE_DIR}/config/pipeline.yaml"
    EE_VERSION="${PROJECT_VERSION}")

if(EE_WITH_DESKTOP)
    add_executable(emotionedge-desktop
        desktop/main.cpp
        desktop/desktop_session.cpp
        desktop/png_writer.cpp
        desktop/shaped_text.cpp
        desktop/stb_truetype_impl.cpp)
    target_link_libraries(emotionedge-desktop PRIVATE ee::pipeline ee::imgui harfbuzz ee_project_options)
    target_compile_definitions(emotionedge-desktop PRIVATE EE_DESKTOP_FONT="${EE_DESKTOP_FONT}")
    # Third-party code compiled here (stb_truetype as ImGui bundles it): not held to our warnings.
    set_source_files_properties(desktop/stb_truetype_impl.cpp PROPERTIES
        COMPILE_OPTIONS $<IF:$<CXX_COMPILER_ID:MSVC>,/w,-w>)
endif()
