cmake_minimum_required(VERSION 3.14)
project(TrunkLPR LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# PCL
find_package(PCL 1.12 REQUIRED COMPONENTS common io kdtree search segmentation)
# yaml-cpp
find_package(yaml-cpp REQUIRED)
# OpenMP (optional)
find_package(OpenMP)

add_executable(trunk_lpr
    src/main.cpp
    src/intra_eval.cpp
    src/inter_eval.cpp
    src/trunk_extractor.cpp
    src/evaluator.cpp
    src/spectrum_descriptor.cpp
    src/local_matcher.cpp
    src/pose_loader.cpp
)

target_compile_definitions(trunk_lpr PRIVATE ${PCL_DEFINITIONS})
target_include_directories(trunk_lpr PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PCL_INCLUDE_DIRS}
)
target_link_directories(trunk_lpr PRIVATE ${PCL_LIBRARY_DIRS})
target_link_libraries(trunk_lpr
    ${PCL_LIBRARIES}
    yaml-cpp
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(trunk_lpr OpenMP::OpenMP_CXX)
endif()
