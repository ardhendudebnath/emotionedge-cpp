# Core third-party dependencies: nlohmann/json (manifests, session export) and yaml-cpp (pipeline
# graph). With a vcpkg toolchain (see vcpkg.json) find_package() satisfies them; otherwise
# FetchContent downloads the pinned, hash-verified release archives.
include(FetchContent)
set(FETCHCONTENT_QUIET ON)

find_package(Threads REQUIRED)

if(EE_FETCH_DEPS)
    FetchContent_Declare(nlohmann_json
        URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
        URL_HASH SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa
        FIND_PACKAGE_ARGS 3.11 CONFIG)

    set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_FORMAT_SOURCE OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(yaml-cpp
        URL https://github.com/jbeder/yaml-cpp/releases/download/yaml-cpp-0.9.0/yaml-cpp-yaml-cpp-0.9.0.tar.gz
        URL_HASH SHA256=298593d9c440fd9034b8b193d96318b76d49bc97c6ceadb7b0836edf0b6d7539
        FIND_PACKAGE_ARGS 0.7 CONFIG)

    FetchContent_MakeAvailable(nlohmann_json yaml-cpp)
else()
    find_package(nlohmann_json 3.11 CONFIG REQUIRED)
    find_package(yaml-cpp CONFIG REQUIRED)
endif()

# Older yaml-cpp installs export a plain `yaml-cpp` target instead of the namespaced one.
if(NOT TARGET yaml-cpp::yaml-cpp AND TARGET yaml-cpp)
    add_library(yaml-cpp::yaml-cpp ALIAS yaml-cpp)
endif()
