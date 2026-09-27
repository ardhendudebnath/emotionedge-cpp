# bench/: the latency harness that validates the blueprint's budget (p.2) and success targets.
add_executable(ee_bench latency_bench.cpp)
target_link_libraries(ee_bench PRIVATE ee::pipeline ee_project_options nlohmann_json::nlohmann_json)
target_compile_definitions(ee_bench PRIVATE EE_DEFAULT_CONFIG="${PROJECT_SOURCE_DIR}/config/pipeline.yaml")
if(WIN32)
    target_link_libraries(ee_bench PRIVATE psapi)
endif()
