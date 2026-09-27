# Project-wide compiler settings shared by every EmotionEdge target (not applied to third-party code).
add_library(ee_project_options INTERFACE)
add_library(ee::project_options ALIAS ee_project_options)

target_compile_features(ee_project_options INTERFACE cxx_std_20)

if(MSVC)
    # C4324: padding from alignas() is intentional (cache-line separated queue indices).
    target_compile_options(ee_project_options INTERFACE /W4 /permissive- /Zc:__cplusplus /utf-8 /EHsc /wd4324)
    target_compile_definitions(ee_project_options INTERFACE NOMINMAX WIN32_LEAN_AND_MEAN _USE_MATH_DEFINES)
    if(EE_WARNINGS_AS_ERRORS)
        target_compile_options(ee_project_options INTERFACE /WX)
    endif()
else()
    target_compile_options(ee_project_options INTERFACE
        -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor -Woverloaded-virtual
        -Wimplicit-fallthrough -Wformat=2)
    if(EE_WARNINGS_AS_ERRORS)
        target_compile_options(ee_project_options INTERFACE -Werror)
    endif()
endif()
