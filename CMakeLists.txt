cmake_minimum_required(VERSION 3.18)

project(CudaAcoustics LANGUAGES CXX CUDA)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

set(SOURCES
    src/main.cpp
    src/gpu/raygen.cu
    src/gpu/intersections.cuh
)

add_executable(AcousticSim ${SOURCES})

target_include_directories(AcousticSim PRIVATE include)

target_link_libraries(AcousticSim PRIVATE CUDA::cudart)