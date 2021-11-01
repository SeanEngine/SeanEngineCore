cmake_minimum_required(VERSION 3.19)
project(CudaNNGen2 CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(CudaNNGen2 main.cu utils/Matrix.cu utils/Matrix.cuh utils/logger.cu utils/logger.cuh layers/Layer.cu layers/Layer.cuh layers/DenseLayer.cu layers/DenseLayer.cuh)
target_link_libraries(CudaNNGen2 cublas.lib)

set_target_properties(
        CudaNNGen2
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)