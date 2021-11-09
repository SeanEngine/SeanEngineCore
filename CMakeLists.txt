cmake_minimum_required(VERSION 3.19)
project(CudaNNGen2 CUDA CXX)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_executable(CudaNNGen2 main.cu utils/Matrix.cu utils/Matrix.cuh utils/logger.cu utils/logger.cuh layers/Layer.cu
        layers/Layer.cuh layers/DenseLayer.cu layers/DenseLayer.cuh execution/ThreadControls.cpp execution/ThreadControls.h)
target_link_libraries(CudaNNGen2 cublas.lib)
target_link_libraries(CudaNNGen2 Threads::Threads)

set_target_properties(
        CudaNNGen2
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)