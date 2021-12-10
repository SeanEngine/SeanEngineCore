//
// Created by DanielSun on 12/8/2021.
//

#include "NeuralUtils.cuh"
#include <cooperative_groups.h>
#include <cstdio>
#include <iostream>
//this will trigger an exception if the condition do not met
#define cuAssert(condition) if(!(condition)){ asm{"trap"}; }

__inline__ __device__ float warpReduce(float val) {
    for (int mask = WARP_SIZE >> 1; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

//this thing only process matrices below the size of 1024 elements
//since nvidia decided not to code in their driver a way to sync all blocks
__global__ void softmax1024(int n, const float* src, float* dist){
    unsigned int globalID = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int laneID = globalID % WARP_SIZE;
    __shared__ float buffer[CUDA_SOFTMAX_BLOCK];

    float value = globalID < n ? exp(src[globalID]) : 0;
    float reduceValue;
    buffer[globalID] = value;
    __syncthreads();

    int procSize = n;

    //cross warp reductions together with warp reduction
    while(procSize/WARP_SIZE > 0){
         reduceValue = globalID < procSize ? buffer[globalID] : 0;
         __syncthreads();
         reduceValue = warpReduce(reduceValue);
         if(laneID == 0 && globalID < procSize) buffer[globalID/WARP_SIZE] = reduceValue;
         procSize = procSize%WARP_SIZE ? procSize/WARP_SIZE + 1 : procSize/WARP_SIZE;
         __syncthreads();
    }

    //the last iteration
    reduceValue = globalID < procSize ? buffer[globalID] : 0;
    __syncthreads();
    reduceValue = warpReduce(reduceValue);
    if(laneID == 0 && globalID < procSize ) buffer[globalID/WARP_SIZE] = reduceValue;
    __syncthreads();

    dist[globalID] = value / buffer[0];
}

//store every exponents in the buffer
__global__ void softMaxPrepare(int n, float* buffer){
    unsigned int globalID = threadIdx.x + blockIdx.x * blockDim.x;
    if(globalID < n) buffer[globalID] = exp(buffer[globalID]);
}

//execute reduction like normally
__global__ void softMaxReduce(int n, float* buffer){
    int globalID = threadIdx.x + blockIdx.x * blockDim.x;
    int warpID = globalID % WARP_SIZE;
    float val = globalID < n ? buffer[globalID] : 0;
    __syncthreads();
    warpReduce(val);
    if(warpID == 0) buffer[globalID/WARP_SIZE] = val;
}

//use the result on the elements
__global__ void softMaxActivate(int n, const float* buffer, float* dist){
    unsigned int globalID = threadIdx.x + blockIdx.x * blockDim.x;
    if(globalID < n)
    dist[globalID] = exp(dist[globalID]) / buffer[0];
}

__global__ void sigmoidActivation(Matrix::Matrix2d *mat1) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float x = mat1->get(row, col);
    mat1->set(row, col, 1.0f / (1.0f + exp(-x)));
}

__device__ float sigmoidCalc(float x) {
    return 1.0f / (1.0f + exp(-x));
}

__global__ void sigmoidActivation(Matrix::Matrix2d *mat1, Matrix::Matrix2d *result) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float x = mat1->get(row, col);
    result->set(row, col, sigmoidCalc(x));
}

__global__ void sigmoidDerivative(Matrix::Matrix2d *mat1, Matrix::Matrix2d *result) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float x = mat1->get(row, col);
    result->set(row, col, sigmoidCalc(x) * (1.0f - sigmoidCalc(x)));
}


__global__ void leakyReluActivation(Matrix::Matrix2d *mat1, Matrix::Matrix2d *result, float ALPHA) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float x = mat1->get(row, col);
    result->set(row, col, x > 0 ? x : ALPHA * x);
}

__global__ void leakyReluDerivative(Matrix::Matrix2d *mat1, Matrix::Matrix2d *result, float ALPHA) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float x = mat1->get(row, col);
    result->set(row, col, x > 0 ? 1 : ALPHA);
}

//activations
Matrix::Matrix2d *NeuralUtils::callActivationSigmoid(Matrix::Matrix2d *mat1) {
    dim3 gridSize = dim3((mat1->colcount + CUDA_BLOCK_SIZE.x - 1) / (CUDA_BLOCK_SIZE.x),
                         (mat1->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    sigmoidActivation<<<gridSize, CUDA_BLOCK_SIZE>>>(mat1);
    cudaDeviceSynchronize();
    return mat1;
}

Matrix::Matrix2d *NeuralUtils::callActivationSigmoid(Matrix::Matrix2d *mat1, Matrix::Matrix2d *result) {
    dim3 gridSize = dim3((mat1->colcount + CUDA_BLOCK_SIZE.x - 1) / (CUDA_BLOCK_SIZE.x),
                         (mat1->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    sigmoidActivation<<<gridSize, CUDA_BLOCK_SIZE>>>(mat1, result);
    cudaDeviceSynchronize();
    return result;
}

Matrix::Matrix2d *NeuralUtils::callDerivativeSigmoid(Matrix::Matrix2d *mat1, Matrix::Matrix2d *result) {
    dim3 gridSize = dim3((mat1->colcount + CUDA_BLOCK_SIZE.x - 1) / (CUDA_BLOCK_SIZE.x),
                         (mat1->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    sigmoidDerivative<<<gridSize, CUDA_BLOCK_SIZE>>>(mat1, result);
    cudaDeviceSynchronize();
    return result;
}

Matrix::Matrix2d *NeuralUtils::callLeakyReluDerivative(Matrix::Matrix2d *mat1, Matrix::Matrix2d *result, float ALPHA) {
    dim3 gridSize = dim3((mat1->colcount + CUDA_BLOCK_SIZE.x - 1) / (CUDA_BLOCK_SIZE.x),
                         (mat1->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    leakyReluDerivative<<<gridSize, CUDA_BLOCK_SIZE>>>(mat1, result, ALPHA);
    cudaDeviceSynchronize();
    return result;
}

Matrix::Matrix2d *NeuralUtils::callLeakyReluActivation(Matrix::Matrix2d *mat1, Matrix::Matrix2d *result, float ALPHA) {
    dim3 gridSize = dim3((mat1->colcount + CUDA_BLOCK_SIZE.x - 1) / (CUDA_BLOCK_SIZE.x),
                         (mat1->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    leakyReluActivation<<<gridSize, CUDA_BLOCK_SIZE>>>(mat1, result, ALPHA);
    cudaDeviceSynchronize();
    return result;
}

//buffer can be set to null if the softmax operation is applied to matrices less than 1024 elements
//call the softmax activation
Matrix::Matrix2d *NeuralUtils::callSoftMax(Matrix::Matrix2d *mat1, Matrix::Matrix2d *result, float* buffer) {
    assert(mat1->rowcount * mat1->colcount == result->rowcount * result->colcount);
    int n =  mat1->rowcount * mat1->colcount;
    unsigned int gridSize = n/ (CUDA_BLOCK_SIZE.x * CUDA_BLOCK_SIZE.y) + 1;
    unsigned int blockSize = CUDA_BLOCK_SIZE.x * CUDA_BLOCK_SIZE.y;
    if(n <= 1024) {
        softmax1024<<<1, CUDA_SOFTMAX_BLOCK>>>(n, mat1->elements, result->elements);
        cudaDeviceSynchronize();
        return result;
    }
    assert(buffer != nullptr);
    cudaMemcpy(buffer, mat1->elements, sizeof(float) *n, cudaMemcpyDeviceToDevice);
    softMaxPrepare<<<gridSize, blockSize>>>(n, buffer);
    cudaDeviceSynchronize();
    int procSize = n;
    while(procSize/WARP_SIZE > 0){
        softMaxReduce<<<gridSize, blockSize>>>(procSize, buffer);
        procSize = procSize%WARP_SIZE ? procSize/WARP_SIZE + 1 : procSize/WARP_SIZE;
        cudaDeviceSynchronize();
    }
    softMaxReduce<<<gridSize,blockSize>>>(procSize, buffer);
    cudaDeviceSynchronize();
    softMaxActivate<<<gridSize, blockSize>>>(n, buffer, result->elements);
    cudaDeviceSynchronize();
    return result;
}
