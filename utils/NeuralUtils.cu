//
// Created by DanielSun on 12/8/2021.
//

#include "NeuralUtils.cuh"
#include <cooperative_groups.h>
#include <cstdio>
#include <iostream>

__global__ void allocConvBias(Matrix::Matrix2d* outputBuffer, Matrix::Matrix2d* biases){
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    outputBuffer->set(row, col, biases->get(row,0));
}

__inline__ __device__ float warpReduce(float val) {
    #pragma unroll
    for (int mask = WARP_SIZE >> 1; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

__inline__ __device__ float warpCompare(float val) {
    #pragma unroll
    for (int mask = WARP_SIZE >> 1; mask > 0; mask >>= 1) {
        float temp = __shfl_xor_sync(0xffffffff, val, mask);
        val = temp > val ? temp : val;
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

    unsigned int procSize = n;

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

    if(globalID<n)
    dist[globalID] = value / buffer[0];
}

// this method will divide all elements of the matrix by the largest element
// preventing issues caused by overflowing of 32-bit floats with increasing model size.
__global__ void softmaxControlled1024(int n, const float* src, float* dist){
    unsigned int globalID = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int laneID = globalID % WARP_SIZE;
    __shared__ float buffer[CUDA_SOFTMAX_BLOCK];
    float value = globalID < n ? src[globalID] : 0;
    float reduceValue;
    buffer[globalID] = value;
    __syncthreads();

    unsigned int procSize = n;

    //run the reduction but for the max value
    while(procSize/WARP_SIZE > 0){
        reduceValue = globalID < procSize ? buffer[globalID] : 0;
        __syncthreads();
        reduceValue = warpCompare(reduceValue);
        if(laneID == 0 && globalID < procSize) buffer[globalID/WARP_SIZE] = reduceValue;
        procSize = procSize%WARP_SIZE ? procSize/WARP_SIZE + 1 : procSize/WARP_SIZE;
        __syncthreads();
    }

    //the last iteration
    reduceValue = globalID < procSize ? buffer[globalID] : 0;
    __syncthreads();
    reduceValue = warpCompare(reduceValue);
    if(laneID == 0 && globalID < procSize ) buffer[globalID/WARP_SIZE] = reduceValue;
    __syncthreads();

    float MAX_VALUE = buffer[0];
    value = globalID < n ? exp(value - MAX_VALUE) : 0;
    buffer[globalID] = value;
    __syncthreads();

    procSize = n;

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

    if(globalID<n)
        dist[globalID] = value / buffer[0];
}

//store every exponents in the buffer
__global__ void softMaxPrepare(unsigned int n, float* buffer){
    unsigned int globalID = threadIdx.x + blockIdx.x * blockDim.x;
    if(globalID < n) buffer[globalID] = exp(buffer[globalID]);
}

//execute reduction like normally
__global__ void softMaxReduce(unsigned int n, float* buffer){
    unsigned int globalID = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int warpID = globalID % WARP_SIZE;
    float val = globalID < n ? buffer[globalID] : 0;
    __syncthreads();
    warpReduce(val);
    if(warpID == 0) buffer[globalID/WARP_SIZE] = val;
}

//use the result on the elements
__global__ void softMaxActivate(unsigned int n, const float* buffer, float* dist){
    unsigned int globalID = threadIdx.x + blockIdx.x * blockDim.x;
    if(globalID < n)
    dist[globalID] = exp(dist[globalID]) / buffer[0];
}

__global__ void softMaxDerivative(Matrix::Matrix2d* mat1, Matrix::Matrix2d* correctOut, Matrix::Matrix2d* result){
    unsigned int globalID = threadIdx.x + blockIdx.x * blockDim.x;
    result->set( globalID, 0,mat1->get(globalID,0) - correctOut->get(globalID,0));
}

// L = - y * ln(a)
__global__ void softMaxCost(Matrix::Matrix2d* mat1, Matrix::Matrix2d* correctOut, Matrix::Matrix2d* result){
    unsigned int globalID = threadIdx.x + blockIdx.x * blockDim.x;
    if (mat1->get(globalID,0) <= 0) mat1->set(globalID, 0, 1e-30);
    result->set( globalID, 0,-(correctOut->get(globalID,0) * log(mat1->get(globalID,0))));
}

__global__ void sigmoidActivation(Matrix::Matrix2d *mat1) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    float x = mat1->get(row, col);
    mat1->set(row, col, 1.0f / (1.0f + exp(-x)));
}

__device__ float sigmoidCalc(float x) {
    return 1.0f / (1.0f + exp(-x));
}

__global__ void sigmoidActivation(Matrix::Matrix2d *mat1, Matrix::Matrix2d *result) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    float x = mat1->get(row, col);
    result->set(row, col, sigmoidCalc(x));
}

__global__ void sigmoidDerivative(Matrix::Matrix2d *mat1, Matrix::Matrix2d *result) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    float x = mat1->get(row, col);
    result->set(row, col, sigmoidCalc(x) * (1.0f - sigmoidCalc(x)));
}

__global__ void leakyReluActivation(Matrix::Matrix2d *mat1, Matrix::Matrix2d *result, float ALPHA) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    float x = mat1->get(row, col);
    result->set(row, col, x > 0 ? x : ALPHA * x);
}

__global__ void leakyReluDerivative(Matrix::Matrix2d *mat1, Matrix::Matrix2d *result, float ALPHA) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    float x = mat1->get(row, col);
    result->set(row, col, x > 0 ? 1 : ALPHA);
}

__global__ void leakyReluActivation(Matrix::Tensor* mat1, Matrix::Tensor* result, float ALPHA){
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < result->elementCount) {
        float x = mat1->elements[index];
        result->elements[index] = x > 0 ? x : ALPHA * x;
    }
}

__global__ void leakyReluDerivative(Matrix::Tensor* mat1, Matrix::Tensor* result, float ALPHA){
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < result->elementCount){
        float x = mat1->elements[index];
        result->elements[index] = x > 0 ? 1 : ALPHA;
    }
}

//this method generates a patched form of the feature maps
//see graph 3 on : https://sahnimanas.github.io/post/anatomy-of-a-high-performance-convolution/
__global__ void imgToCol(Matrix::Tensor3d* featureMaps, Matrix::Matrix2d* featureBuffer, unsigned int outputHeight,
                         unsigned int filterSize, unsigned int stride){
    unsigned int colID = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int depth = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k2 = filterSize*filterSize;
    if(depth < (featureBuffer->rowcount / (k2))) {
        unsigned int startRow = (colID/outputHeight)*stride;
        unsigned int startCol = (colID%outputHeight)*stride;

        #pragma unroll
        for(int i=0; i<filterSize; i++){
            #pragma unroll
            for(int j=0; j<filterSize; j++){
                featureBuffer->set(depth*k2 + i*filterSize + j, colID, featureMaps->get(depth, startRow+i, startCol+j));
            }
        }
    }
}

//this is used in the back propagation
__global__ void colToImg2DNoPadding(Matrix::Matrix2d* errors, Matrix::Matrix2d* propaBuffer, unsigned int outputHeight,
                                    unsigned int sourceHeight, unsigned int filterSize, unsigned int stride, unsigned int padSize){
    unsigned int colID = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int depth = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k2 = filterSize*filterSize;
    unsigned int sourceCols = errors->colcount/sourceHeight;
    if(depth < (propaBuffer->rowcount / (k2))) {
        unsigned int startRow = (colID/outputHeight)*stride;
        unsigned int startCol = (colID%outputHeight)*stride;
        if(startRow< padSize || startCol < padSize || startRow > padSize+sourceHeight || startCol > sourceCols)
            return;

        #pragma unroll
        for(int i=0; i<filterSize; i++){
            #pragma unroll
            for(int j=0; j<filterSize; j++){
                errors->atomAdd(depth, (startRow+i-padSize)*sourceHeight + startCol + j - padSize,
                                propaBuffer->get(depth*k2 + i*filterSize + j, colID));
            }
        }
    }
}

__global__ void pad3d(Matrix::Tensor3d* input, Matrix::Tensor3d* output, unsigned int padSize){
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int depth = blockIdx.z * blockDim.z + threadIdx.z;
    output->set(depth+padSize, row+padSize, col+padSize, input->get(depth, row, col));
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

Matrix::Matrix2d *NeuralUtils::callDerivativeLeakyRelu(Matrix::Matrix2d *mat1, Matrix::Matrix2d *result, float ALPHA) {
    dim3 gridSize = dim3((mat1->colcount + CUDA_BLOCK_SIZE.x - 1) / (CUDA_BLOCK_SIZE.x),
                         (mat1->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    leakyReluDerivative<<<gridSize, CUDA_BLOCK_SIZE>>>(mat1, result, ALPHA);
    cudaDeviceSynchronize();
    return result;
}

Matrix::Matrix2d *NeuralUtils::callActivationLeakyRelu(Matrix::Matrix2d *mat1, Matrix::Matrix2d *result, float ALPHA) {
    dim3 gridSize = dim3((mat1->colcount + CUDA_BLOCK_SIZE.x - 1) / (CUDA_BLOCK_SIZE.x),
                         (mat1->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    leakyReluActivation<<<gridSize, CUDA_BLOCK_SIZE>>>(mat1, result, ALPHA);
    cudaDeviceSynchronize();
    return result;
}

Matrix::Tensor *NeuralUtils::callActivationLeakyRelu(Matrix::Tensor *mat1, Matrix::Tensor *result, float ALPHA) {
    return nullptr;
}

Matrix::Tensor *NeuralUtils::callDerivativeLeakyRelu(Matrix::Tensor *mat1, Matrix::Tensor *result, float ALPHA) {
    return nullptr;
}


//buffer can be set to null if the softmax operation is applied to matrices less than 1024 elements
//call the softmax activation
Matrix::Matrix2d *NeuralUtils::callSoftMax(Matrix::Matrix2d *mat1, Matrix::Matrix2d *result, float* buffer) {
    assert(mat1->rowcount * mat1->colcount == result->rowcount * result->colcount);
    unsigned int n =  mat1->rowcount * mat1->colcount;
    unsigned int gridSize = n/ (CUDA_BLOCK_SIZE.x * CUDA_BLOCK_SIZE.y) + 1;
    unsigned int blockSize = CUDA_BLOCK_SIZE.x * CUDA_BLOCK_SIZE.y;
    if(n <= 1024) {
        softmaxControlled1024<<<1, CUDA_SOFTMAX_BLOCK>>>(n, mat1->elements, result->elements);
        cudaDeviceSynchronize();
        return result;
    }
    assert(buffer != nullptr);
    cudaMemcpy(buffer, mat1->elements, sizeof(float) *n, cudaMemcpyDeviceToDevice);
    softMaxPrepare<<<gridSize, blockSize>>>(n, buffer);
    cudaDeviceSynchronize();
    unsigned int procSize = n;
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

Matrix::Matrix2d *
NeuralUtils::callSoftMaxDerivatives(Matrix::Matrix2d *mat1, Matrix::Matrix2d *correctOut, Matrix::Matrix2d *result) {
    assert(mat1->rowcount == correctOut->rowcount && mat1->rowcount == result->rowcount);
    assert(mat1->colcount == 1 && result->colcount == 1 && correctOut->colcount==1);
    unsigned int n =  mat1->rowcount;
    unsigned int gridSize = n/ (CUDA_BLOCK_SIZE.x * CUDA_BLOCK_SIZE.y) + 1;
    unsigned int blockSize = CUDA_BLOCK_SIZE.x * CUDA_BLOCK_SIZE.y;
    softMaxDerivative<<<gridSize, blockSize>>>(mat1, correctOut, result);
    cudaDeviceSynchronize();
    return result;
}

Matrix::Matrix2d *NeuralUtils::callSoftMaxCost(Matrix::Matrix2d *mat1,Matrix::Matrix2d *correctOut, Matrix::Matrix2d *result) {
    unsigned int n =  mat1->rowcount;
    unsigned int gridSize = n/ (CUDA_BLOCK_SIZE.x * CUDA_BLOCK_SIZE.y) + 1;
    unsigned int blockSize = CUDA_BLOCK_SIZE.x * CUDA_BLOCK_SIZE.y;
    softMaxCost<<<gridSize, blockSize>>>(mat1, correctOut, result);
    cudaDeviceSynchronize();
    return result;
}
// filter: rowcount = colcount
// see : https://sahnimanas.github.io/post/anatomy-of-a-high-performance-convolution/
//outputDim = (n-f)/s + 1
Matrix::Tensor3d *
NeuralUtils::callConv2d(Matrix::Tensor3d *mat1, Matrix::Tensor4d *filter, Matrix::Tensor3d *result, unsigned int stride,
                        Matrix::Matrix2d* filterBuffer, Matrix::Matrix2d* featureBuffer, Matrix::Matrix2d* outputBuffer,
                        Matrix::Matrix2d* biases) {
    //condition check
    assert(mat1->rowcount == mat1->colcount && filter->rowcount==filter->colcount);
    assert(filter->depthCount == mat1->depthCount);
    assert((mat1->rowcount-filter->rowcount) % stride == 0);
    assert(result->rowcount == (mat1->rowcount-filter->rowcount) / stride + 1);
    assert(result->colcount == result->rowcount && result->depthCount == filter->wCount);

    assert(featureBuffer->rowcount == filter->depthCount * filter->rowcount * filter->colcount);
    assert(featureBuffer->colcount == result->rowcount* result->colcount);

    filterBuffer->rowcount = filter->wCount;
    filterBuffer->colcount = filter->depthCount * filter->rowcount * filter->colcount;
    filterBuffer->elements = filter->elements;

    outputBuffer->rowcount = result->depthCount;
    outputBuffer->colcount = result->rowcount * result->colcount;
    outputBuffer->elements = result->elements;

    cudaDeviceSynchronize();
    img2col(mat1, featureBuffer, result->rowcount, filter->rowcount, stride);

    //cross
    crossA(filterBuffer, featureBuffer, callAllocConvBias(outputBuffer, biases));
    cudaDeviceSynchronize();
    return result;
}

//pad zeros around the target
Matrix::Tensor3d *NeuralUtils::padding3d(Matrix::Tensor3d *mat1, Matrix::Tensor3d *output, unsigned int padSize) {
    //check condition
    Matrix::callAllocZero(output);
    assert(output->rowcount == mat1->rowcount + 2*padSize && output->colcount == mat1->colcount + 2*padSize);
    dim3 gridSize = dim3((mat1->colcount + CUDA_BLOCK_SIZE_3D.x -1)/CUDA_BLOCK_SIZE_3D.x
                        ,(mat1->rowcount + CUDA_BLOCK_SIZE_3D.y -1)/CUDA_BLOCK_SIZE_3D.y
                        ,(mat1->depthCount + CUDA_BLOCK_SIZE_3D.z -1)/CUDA_BLOCK_SIZE_3D.z);
    pad3d<<<gridSize, CUDA_BLOCK_SIZE_3D>>>(mat1,output,padSize);
    cudaDeviceSynchronize();
    return output;
}

Matrix::Matrix2d *NeuralUtils::callAllocConvBias(Matrix::Matrix2d *outputBuffer, Matrix::Matrix2d *bias) {
    assert(bias->rowcount == outputBuffer->rowcount && bias->colcount == 1);
    dim3 gridSize = dim3((outputBuffer->colcount + CUDA_BLOCK_SIZE.x-1)/CUDA_BLOCK_SIZE.x,
                         (outputBuffer->rowcount + CUDA_BLOCK_SIZE.y-1)/CUDA_BLOCK_SIZE.y);
    allocConvBias<<<gridSize, CUDA_BLOCK_SIZE>>>(outputBuffer, bias);
    cudaDeviceSynchronize();
    return outputBuffer;
}


Matrix::Matrix2d *
NeuralUtils::callImg2Col(Matrix::Tensor3d *featureMaps, Matrix::Matrix2d *featureBuffer, unsigned int outputHeight,
                         unsigned int filterSize, unsigned int stride) {
    //prepare buffer for gemm operation
    dim3 featureGridSize = dim3((featureBuffer->colcount + CUDA_BLOCK_SIZE.x-1)/CUDA_BLOCK_SIZE.x,
                                (featureBuffer->rowcount/(filterSize*filterSize) + CUDA_BLOCK_SIZE.y-1)/ CUDA_BLOCK_SIZE.y);
    imgToCol<<<featureGridSize, CUDA_BLOCK_SIZE>>>(featureMaps, featureBuffer, outputHeight, filterSize, stride);
    cudaDeviceSynchronize();
    return featureBuffer;
}

Matrix::Matrix2d *
NeuralUtils::callCol2Img2dNP(Matrix::Matrix2d *errors, Matrix::Matrix2d *propaBuffer, unsigned int outputHeight,
                             unsigned int sourceHeight, unsigned int filterSize, unsigned int stride,
                             unsigned int padSize) {
    dim3 featureGridSize = dim3((propaBuffer->colcount + CUDA_BLOCK_SIZE.x-1)/CUDA_BLOCK_SIZE.x,
                                (propaBuffer->rowcount/(filterSize*filterSize) + CUDA_BLOCK_SIZE.y-1)/ CUDA_BLOCK_SIZE.y);
    colToImg2DNoPadding<<<featureGridSize, CUDA_BLOCK_SIZE>>>(errors, propaBuffer, outputHeight, sourceHeight,
                                                              filterSize, stride, padSize);
    cudaDeviceSynchronize();
    return errors;
}
