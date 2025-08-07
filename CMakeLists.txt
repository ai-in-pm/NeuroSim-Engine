cmake_minimum_required(VERSION 3.16)
project(NeuroSimEngine VERSION 1.0.0 LANGUAGES CXX)

# Project Information
# Owner: Darrell Mesa (darrell.mesa@pm-ss.org)
# GitHub: https://github.com/ai-in-pm
# Description: Neural simulation engine for autism and PTSD modeling

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler-specific options
if(MSVC)
    add_compile_options(/W4 /permissive-)
    add_compile_definitions(_USE_MATH_DEFINES)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find required packages (optional for initial build)
find_package(Eigen3 QUIET)
find_package(nlohmann_json QUIET)

# Find Python and pybind11 for Python bindings (optional)
find_package(Python3 COMPONENTS Interpreter Development QUIET)
find_package(pybind11 QUIET)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/core)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/regions)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inputs)

# Core library sources
set(CORE_SOURCES
    core/simulator.cpp
    core/brain_router.cpp
    core/multimodal_fusion.cpp
    core/memory_overlay.cpp
    core/flashback_overlay.cpp
)

# Region model sources
set(REGION_SOURCES
    regions/amygdala.cpp
    regions/hippocampus.cpp
    regions/insula.cpp
    regions/prefrontal.cpp
    regions/cerebellum.cpp
    regions/microcircuit.cpp
)

# Input processing sources
set(INPUT_SOURCES
    inputs/image_to_embedding.cpp
    inputs/audio_to_embedding.cpp
    inputs/vestibular_synth.cpp
    inputs/interoceptive_sim.cpp
)

# Create the main NeuroSim library
add_library(neurosim_core STATIC
    ${CORE_SOURCES}
    ${REGION_SOURCES}
    ${INPUT_SOURCES}
)

# Link libraries (conditional)
if(Eigen3_FOUND)
    target_link_libraries(neurosim_core Eigen3::Eigen)
    target_compile_definitions(neurosim_core PRIVATE HAVE_EIGEN3)
endif()

if(nlohmann_json_FOUND)
    target_link_libraries(neurosim_core nlohmann_json::nlohmann_json)
    target_compile_definitions(neurosim_core PRIVATE HAVE_NLOHMANN_JSON)
endif()

# Create Python bindings (conditional)
if(pybind11_FOUND AND Python3_FOUND)
    pybind11_add_module(neurosim_py pybind/neuro_api_bindings.cpp)
    target_link_libraries(neurosim_py PRIVATE neurosim_core)

    # Compiler-specific definitions for pybind11
    target_compile_definitions(neurosim_py PRIVATE VERSION_INFO="1.0.0")
endif()

# Create simple test executable (no dependencies required)
add_executable(simple_test test/simple_test.cpp)

# Create full test executable (requires dependencies)
if(Eigen3_FOUND AND nlohmann_json_FOUND)
    add_executable(neurosim_test
        test/test_basic_simulation.cpp
    )
    target_link_libraries(neurosim_test neurosim_core)
endif()

# Installation (conditional)
install(TARGETS neurosim_core DESTINATION lib)
install(TARGETS simple_test DESTINATION bin)

if(TARGET neurosim_py)
    install(TARGETS neurosim_py DESTINATION .)
endif()

if(TARGET neurosim_test)
    install(TARGETS neurosim_test DESTINATION bin)
endif()

install(DIRECTORY core/ DESTINATION include/neurosim/core FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY regions/ DESTINATION include/neurosim/regions FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY inputs/ DESTINATION include/neurosim/inputs FILES_MATCHING PATTERN "*.hpp")

# Enable testing
enable_testing()
add_test(NAME simple_neurosim_test COMMAND simple_test)

if(TARGET neurosim_test)
    add_test(NAME neurosim_unit_tests COMMAND neurosim_test)
endif()

# Package configuration
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
