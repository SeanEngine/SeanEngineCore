//
// Created by DanielSun on 9/23/2021.
//

#include "Matrix.cuh"
#include <string>
#include <iostream>
#include <cassert>
#include <windows.h>
#include <profileapi.h>
#include <cuda_runtime_api.h>
#include <mma.h>

class fragment;

using namespace std;
using namespace nvcuda;
//device methods
__device__ float Matrix::Matrix2d::get(unsigned int row, unsigned int col) const {
    return (row < rowcount && col < colcount) ? this->elements[row * this->colcount + col] : 0.0f;
}

__device__ void Matrix::Matrix2d::set(unsigned int row, unsigned int col, float value) const {
    if (row < rowcount && col < colcount)
        this->elements[row * this->colcount + col] = value;
}

__device__ void Matrix::Matrix2d::add(unsigned int row, unsigned int col, float value) const {
    if (row < rowcount && col < colcount)
        this->elements[row * this->colcount + col] += value;
}

__device__ void Matrix::Matrix2d::atomAdd(unsigned int row, unsigned int col, float value) const {
    if (row < rowcount && col < colcount && row >= 0 && col >= 0)
        atomicAdd(this->elements + row * colcount + col, value);
}

__device__ float Matrix::Tensor3d::get(unsigned int depth, unsigned int row, unsigned int col) const {
    if(row >= rowcount && col >= colcount && depth >= depthCount) return 0.0f;
    return this->elements[depth * this->rowcount * this->colcount + row * this->colcount + col];
}

__device__ float Matrix::Tensor3d::get(unsigned int depth, unsigned int offset) const {
    if(offset >= rowcount * colcount && depth >= depthCount) return 0.0f;
    return this->elements[depth * this->rowcount * this->colcount + offset];
}

__device__ void Matrix::Tensor3d::set(unsigned int depth, unsigned int row, unsigned int col, float value) const {
    if(row < rowcount && col < colcount && depth < depthCount)
        this->elements[depth * this->rowcount * this->colcount + row * this->colcount + col] = value;
}

__device__ void Matrix::Tensor3d::set(unsigned int depth, unsigned int offset, float value) const {
    if(offset < rowcount * colcount && depth < depthCount)
         this->elements[depth * this->rowcount * this->colcount + offset] = value;
}

__device__ void Matrix::Tensor3d::add(unsigned int depth, unsigned int row, unsigned int col, float value) const  {
    if(row < rowcount && col < colcount && depth < depthCount)
        this->elements[depth * this->rowcount * this->colcount + row * this->colcount + col] += value;
}

__device__ void Matrix::Tensor3d::atomAdd(unsigned int depth, unsigned int row, unsigned int col, float value) const {
    if(row < rowcount && col < colcount && depth < depthCount)
        atomicAdd(elements + depth * this->rowcount * this->colcount + row * this->colcount + col, value);
}


__device__ float Matrix::Tensor4d::get(unsigned int w, unsigned int depth, unsigned int row, unsigned int col) const {
    if(row >= rowcount && col >= colcount && depth >= depthCount && w > wCount) return 0.0f;
    return this->elements[w*depthCount*rowcount*colcount + depth*rowcount*colcount + row*colcount + col];
}

__device__ void Matrix::Tensor4d::set(unsigned int w, unsigned int depth, unsigned int row, unsigned int col, float value) const {
    if(row < rowcount && col < colcount && depth < depthCount && w < wCount)
         this->elements[w*depthCount*rowcount*colcount + depth*rowcount*colcount + row*colcount + col] = value;
}

__host__ void Matrix::Tensor3d::extract2d(unsigned int depth, Matrix2d* mat) const {
    assert(mat->rowcount == rowcount && mat->colcount == colcount && depth < depthCount);
    cudaMemcpy(mat->elements, elements + depth * rowcount * colcount, sizeof(float) * rowcount * colcount, cudaMemcpyDeviceToDevice);
}

__host__ void Matrix::Tensor3d::emplace2d(unsigned int depth, Matrix::Matrix2d *mat) const {
    assert(mat->rowcount == rowcount && mat->colcount == colcount && depth < depthCount);
    cudaMemcpy(elements + depth * rowcount * colcount, mat->elements, sizeof(float) * rowcount * colcount, cudaMemcpyDeviceToDevice);
}

__host__ string Matrix::Tensor3d::toString() const {
    return "(" + to_string(depthCount) + "," + to_string(rowcount) + "," + to_string(colcount) + ")";
}


__device__ float Matrix::fasterSqrt(float in) {
    float half = 0.5f * in;
    int i = *(int *) &in;
    i = 0x5f375a86 - (i >> 1);
    in = *(float *) &i;
    in = in * (1.5f - half * in * in);
    return in;
}

//random number fill (0-1)
__global__ void allocRandom(long seed, Matrix::Matrix2d *mat1) {
    curandStateXORWOW_t state;
    unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init((row + 1) * (col + 1) * seed, 0, 0, &state);
    mat1->set(row, col, static_cast<float>((curand_uniform(&state)) - 0.5F));
}

__global__ void allocConst(Matrix::Matrix2d *mat1, float in) {
    unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
    mat1->set(row, col, in);
}

__global__ void allocRandom(long seed, Matrix::Tensor *mat1){
    curandStateXORWOW_t state;
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init((index+1) * seed, 0, 0, &state);
    mat1->elements[index] =  static_cast<float>((curand_uniform(&state)) - 0.5F);
}

__global__ void allocConst(Matrix::Tensor* mat1, float in){
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    mat1->elements[index] = in;
}

#define M 16
#define N 16
#define K 8

__global__ void crossWMMA(Matrix::Matrix2d *mat1, Matrix::Matrix2d *mat2, Matrix::Matrix2d *result) {
    //define warps and lanes
    unsigned int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    unsigned int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    wmma::fragment<wmma::matrix_a, M, N, K, wmma::precision::tf32, wmma::row_major> A;
    wmma::fragment<wmma::matrix_b, M, N, K, wmma::precision::tf32, wmma::row_major> B;
    wmma::fragment<wmma::accumulator, M, N, K, float> CAccu;

    wmma::fill_fragment(CAccu, 0.0f);

    for (int kIndex = 0; kIndex < mat1->colcount; kIndex += K) {
        unsigned int mat1Row = warpM * M;
        unsigned int mat1Col = kIndex;
        unsigned int mat2Row = kIndex;
        unsigned int mat2Col = warpN * N;
        if (mat1Row < mat1->rowcount && mat1Col < mat1->colcount && mat2Row < mat2->rowcount &&
            mat2Col < mat2->colcount) {
            wmma::load_matrix_sync(A, mat1->elements + mat1Row * mat1->colcount + mat1Col, mat1->colcount);
            wmma::load_matrix_sync(B, mat2->elements + mat2Row * mat2->colcount + mat2Col, mat2->colcount);

            #pragma unroll
            for (float &t: A.x) {
                t = wmma::__float_to_tf32(t);
            }

            #pragma unroll
            for (float &t: B.x) {
                t = wmma::__float_to_tf32(t);
            }

            wmma::mma_sync(CAccu, A, B, CAccu);
        }
    }

    unsigned int resultRow = warpM * M;
    unsigned int resultCol = warpN * N;
    wmma::store_matrix_sync(result->elements + resultRow * mat2->colcount + resultCol, CAccu, mat2->colcount,
                            wmma::mem_row_major);
}

__global__ void crossTilingWMMA(Matrix::Matrix2d *mat1, Matrix::Matrix2d *mat2, Matrix::Matrix2d *result) {

    __shared__ float tileMat1[M * K * 4];
    __shared__ float tileMat2[K * N * 4];

    wmma::fragment<wmma::matrix_a, M, N, K, wmma::precision::tf32, wmma::row_major> A;
    wmma::fragment<wmma::matrix_b, M, N, K, wmma::precision::tf32, wmma::row_major> B;
    wmma::fragment<wmma::accumulator, M, N, K, float> CAccu;

    wmma::fill_fragment(CAccu, 0.0f);

    unsigned int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    unsigned int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    unsigned int mat1Row = warpM * M;
    unsigned int mat2Col = warpN * N;
    //The block size is 128*4, while the tiles are 16*32,each thread extract an element
    for (int KTileIndex = 0; KTileIndex < mat1->colcount; KTileIndex += (K * 4)) {

        //copy into shapes : 16 * 32 : 32 * 16
        unsigned int copyX1 = threadIdx.x % 32;
        unsigned int copyY1 = threadIdx.y * (threadIdx.x / 32);
        unsigned int copyX2 = threadIdx.x % 16;
        unsigned int copyY2 = threadIdx.y * (threadIdx.x / 16);

        tileMat1[copyY1 * 32 + copyX1] = mat1->get(mat1Row + copyY1, KTileIndex + copyX1);
        tileMat2[copyY2 * 16 + copyX2] = mat2->get(KTileIndex + copyY2, mat2Col + copyX2);

        __syncthreads();

        for (int kIndex = 0; kIndex < 4 * K; kIndex+=K){
            if (mat1Row < mat1->rowcount &&  kIndex + KTileIndex < mat1->colcount && KTileIndex + kIndex < mat2->rowcount &&
                mat2Col < mat2->colcount) {
                wmma::load_matrix_sync(A, tileMat1 + kIndex, 32);
                wmma::load_matrix_sync(B, tileMat2 + kIndex * 16, 16);

                #pragma unroll
                for (float &t: A.x) {
                    t = wmma::__float_to_tf32(t);
                }

                #pragma unroll
                for (float &t: B.x) {
                    t = wmma::__float_to_tf32(t);
                }

                wmma::mma_sync(CAccu, A, B, CAccu);
            }
        }
    }
    unsigned int resultRow = mat1Row;
    unsigned int resultCol = mat2Col;
    wmma::store_matrix_sync(result->elements + resultRow * mat2->colcount + resultCol, CAccu, mat2->colcount,
                            wmma::mem_row_major);
}

__global__ void crossP(Matrix::Matrix2d *mat1, Matrix::Matrix2d *mat2, Matrix::Matrix2d *result) {
    float currentValue = 0.0;
    unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = 0; i < mat1->colcount; ++i) {
        currentValue += mat1->get(row, i) *
                        mat2->get(i, col);
    }
    result->set(row, col, currentValue);
}

/* constants : TILE_SIZE = blockSize.x = blockSize.y,
 * A normal tiling method that uses shared memory to accelerate calculation
 */
__global__ void crossTiling(Matrix::Matrix2d *mat1, Matrix::Matrix2d *mat2, Matrix::Matrix2d *result) {

    __shared__ float mat1_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float mat2_tile[TILE_SIZE][TILE_SIZE];
    float resultOutput = 0;   // result of C in register

    unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;

    #pragma unroll
    for (int tileId = 0; tileId < (mat1->colcount + TILE_SIZE - 1) / TILE_SIZE; tileId++) {

        //load shared memory
        mat1_tile[threadIdx.y][threadIdx.x] = mat1->get(row, threadIdx.x + tileId * TILE_SIZE);
        mat2_tile[threadIdx.y][threadIdx.x] = mat2->get(threadIdx.y + tileId * TILE_SIZE, col);
        __syncthreads();

        #pragma unroll
        for (int mulIndex = 0; mulIndex < TILE_SIZE; mulIndex++) {
            resultOutput += mat1_tile[threadIdx.y][mulIndex] * mat2_tile[mulIndex][threadIdx.x];
        }
        __syncthreads();
    }

    result->set(row, col, resultOutput);
}

/*
 * This method requires the distribution of grid and blocks to be:
 *     dim3 blockSize(TILE_SIZE, VECTOR_SIZE);
 *     dim3 grid(mat2->colcount / (TILE_SIZE * VECTOR_SIZE), mat1->rowcount / TILE_SIZE);
 * This method use registers to accelerate the process while using outer product instead of inner product
 * this will reduce the computation time cost by 1 cycle per multiplication
 */

__global__ void crossCompOpt(Matrix::Matrix2d *mat1, Matrix::Matrix2d *mat2, Matrix::Matrix2d *result) {

    __shared__ float mat1Tile[TILE_SIZE][TILE_SIZE];
    float mat2Value = 0.0f;
    float resultBuffer[TILE_SIZE] = {0};

    unsigned int resultCol = VECTOR_SIZE * TILE_SIZE * blockIdx.x + threadIdx.y * TILE_SIZE + threadIdx.x;

    for (int tileId = 0; tileId < (mat1->colcount + TILE_SIZE - 1) / TILE_SIZE; tileId++) {
        //allocate elements
        for (int i = 0; i < TILE_SIZE / VECTOR_SIZE; i++) {
            //transposeOperation the matrix segment
            mat1Tile[threadIdx.x][i * VECTOR_SIZE + threadIdx.y] = mat1->
                    get(blockIdx.y * TILE_SIZE + i * VECTOR_SIZE + threadIdx.y, tileId * TILE_SIZE + threadIdx.x);
        }
        __syncthreads();

         #pragma unroll
        for (int row = 0; row < TILE_SIZE; row++) {
            //pick a value of mat2 and put it into the registers
            mat2Value = mat2->get(tileId * TILE_SIZE + row, resultCol);
            #pragma unroll
            for (int bufId = 0; bufId < TILE_SIZE; bufId++) {
                resultBuffer[bufId] += mat1Tile[row][bufId] * mat2Value;
            }
        }
        __syncthreads();
    }
    unsigned int resultRow0 = blockIdx.y * TILE_SIZE;
    for (int bufId = 0; bufId < TILE_SIZE; bufId++) {
        result->set(resultRow0 + bufId, resultCol, resultBuffer[bufId]);
    }
}

//this is basically the same as compOpt but with a tile of mat1 prefetched
__global__ void crossPrefetching(Matrix::Matrix2d *mat1, Matrix::Matrix2d *mat2, Matrix::Matrix2d *result) {

    __shared__ float mat1Tile[TILE_SIZE * TILE_SIZE];
    __shared__ float mat1TileNext[TILE_SIZE * TILE_SIZE];
    float mat2Value = 0.0f;
    float resultBuffer[TILE_SIZE] = {0};

    float *cur = mat1Tile;
    float *next = mat1TileNext;

    unsigned int resultCol = VECTOR_SIZE * TILE_SIZE * blockIdx.x + threadIdx.y * TILE_SIZE + threadIdx.x;
    #pragma unroll
    for (int i = 0; i < TILE_SIZE / VECTOR_SIZE; i++) {
        //transposeOperation the matrix segment
        cur[threadIdx.x * TILE_SIZE + i * VECTOR_SIZE + threadIdx.y] = mat1->
                get(blockIdx.y * TILE_SIZE + i * VECTOR_SIZE + threadIdx.y, threadIdx.x);
    }
    __syncthreads();

    for (int tileId = 0; tileId < (mat1->colcount + TILE_SIZE - 1) / TILE_SIZE; tileId++) {
        if ((tileId + 1) * TILE_SIZE < mat1->colcount + TILE_SIZE - 1) {
            #pragma unroll
            for (int i = 0; i < TILE_SIZE / VECTOR_SIZE; i++) {
                //transposeOperation the matrix segment
                next[threadIdx.x * TILE_SIZE + i * VECTOR_SIZE + threadIdx.y] = mat1->
                        get(blockIdx.y * TILE_SIZE + i * VECTOR_SIZE + threadIdx.y, (tileId+1) * TILE_SIZE + threadIdx.x);
            }
        }

        #pragma unroll
        for (int row = 0; row < TILE_SIZE; row++) {
            //pick a value of mat2 and put it into the registers
            mat2Value = mat2->get(tileId * TILE_SIZE + row, resultCol);
            #pragma unroll
            for (int bufId = 0; bufId < TILE_SIZE; bufId++) {
                resultBuffer[bufId] += cur[row * TILE_SIZE + bufId] * mat2Value;
            }
        }
        __syncthreads();

        //swap the pointers
        auto* tmp = cur;
        cur = next;
        next = tmp;
    }
    unsigned int resultRow0 = blockIdx.y * TILE_SIZE;
    #pragma unroll
    for (int bufId = 0; bufId < TILE_SIZE; bufId++) {
        result->set(resultRow0 + bufId, resultCol, resultBuffer[bufId]);
    }
}

//this is basically the same as compOpt but with a tile of mat1 prefetched
__global__ void crossPrefetchingA(Matrix::Matrix2d *mat1, Matrix::Matrix2d *mat2, Matrix::Matrix2d *result) {

    __shared__ float mat1Tile[TILE_SIZE * TILE_SIZE];
    __shared__ float mat1TileNext[TILE_SIZE * TILE_SIZE];
    float mat2Value = 0.0f;
    float resultBuffer[TILE_SIZE] = {0};

    unsigned int resultCol = VECTOR_SIZE * TILE_SIZE * blockIdx.x + threadIdx.y * TILE_SIZE + threadIdx.x;
    #pragma unroll
    for (int i = 0; i < TILE_SIZE / VECTOR_SIZE; i++) {
        //transposeOperation the matrix segment
        mat1Tile[threadIdx.x * TILE_SIZE + i * VECTOR_SIZE + threadIdx.y] = mat1->
                get(blockIdx.y * TILE_SIZE + i * VECTOR_SIZE + threadIdx.y, threadIdx.x);
    }
    __syncthreads();

    float *cur = mat1Tile;
    float *next = mat1TileNext;

    for (int tileId = 0; tileId < (mat1->colcount + TILE_SIZE - 1) / TILE_SIZE; tileId++) {
        if ((tileId + 1) * TILE_SIZE < mat1->colcount) {
            #pragma unroll
            for (int i = 0; i < TILE_SIZE / VECTOR_SIZE; i++) {
                //transposeOperation the matrix segment
                next[threadIdx.x * TILE_SIZE + i * VECTOR_SIZE + threadIdx.y] = mat1->
                        get(blockIdx.y * TILE_SIZE + i * VECTOR_SIZE + threadIdx.y, (tileId+1) * TILE_SIZE + threadIdx.x);
            }
        }

        #pragma unroll
        for (int row = 0; row < TILE_SIZE; row++) {
            //pick a value of mat2 and put it into the registers
            mat2Value = mat2->get(tileId * TILE_SIZE + row, resultCol);
            for (int bufId = 0; bufId < TILE_SIZE; bufId++) {
                resultBuffer[bufId] += cur[row * TILE_SIZE + bufId] * mat2Value;
            }
        }
        __syncthreads();

        //swap the pointers
        auto *tmp = cur;
        cur = next;
        next = tmp;
    }
    unsigned int resultRow0 = blockIdx.y * TILE_SIZE;
    #pragma unroll
    for (int bufId = 0; bufId < TILE_SIZE; bufId++) {
        result->add(resultRow0 + bufId, resultCol, resultBuffer[bufId]);
    }
}


__global__ void reduction(unsigned int n, const float *input, float *result) {

    unsigned int globalID = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int warpID = globalID % WARP_SIZE;
    float val = globalID < n ? input[globalID] : 0;
    __syncthreads();
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        //let all threads to add the value from other threads
        val += __shfl_xor_sync(0xffffffff, val, offset, WARP_SIZE);
    }
    if (warpID == 0) result[globalID / WARP_SIZE] = val;
}

__global__ void hadmardProduct(Matrix::Matrix2d *mat1, Matrix::Matrix2d *mat2) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    mat1->set(row, col, mat1->get(row, col) * mat2->get(row, col));
}

__global__ void constantProduct(Matrix::Matrix2d *mat1, float con) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    mat1->set(row, col, mat1->get(row, col) * con);
}

__global__ void addition(Matrix::Matrix2d *mat1, Matrix::Matrix2d *mat2) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    mat1->set(row, col, mat1->get(row, col) + mat2->get(row, col));
}

__global__ void addition(Matrix::Matrix2d *mat1, float con) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    mat1->set(row, col, mat1->get(row, col) + con);
}

__global__ void subtraction(Matrix::Matrix2d *mat1, Matrix::Matrix2d *mat2) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    mat1->set(row, col, mat1->get(row, col) - mat2->get(row, col));
}

__global__ void subtraction(Matrix::Matrix2d *mat1, float con) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    mat1->set(row, col, mat1->get(row, col) - con);
}

__global__ void power(Matrix::Matrix2d *mat1, float con) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    mat1->set(row, col, pow(mat1->get(row, col), con));
}

__global__ void transposeOperation(Matrix::Matrix2d *mat1, Matrix::Matrix2d *result) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    result->set(col, row, mat1->get(row, col));
}

__global__ void insertColumn(Matrix::Matrix2d *mat1, Matrix::Matrix2d *column, unsigned int colIndex) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    mat1->set(index, colIndex, column->get(index, 0));
}

//memory Control:
// ===============================================================

Matrix::Matrix2d* Matrix::callAllocElementH(unsigned int row, unsigned int col) {
    Matrix2d* mat1;
    cudaMallocHost(&mat1, sizeof(Matrix::Matrix2d));
    mat1->rowcount = row;
    mat1->colcount = col;
    cudaMallocHost(reinterpret_cast<void **>(&mat1->elements), row * col * sizeof(float));
    return mat1;
}

Matrix::Matrix2d* Matrix::callAllocElementD(unsigned int row, unsigned int col) {
    Matrix2d* mat1;
    cudaMallocHost(&mat1, sizeof(Matrix::Matrix2d));
    mat1->rowcount = row;
    mat1->colcount = col;
    cudaMalloc(reinterpret_cast<void **>(&mat1->elements), row * col * sizeof(float));
    return mat1;
}

Matrix::Tensor3d* Matrix::callAllocElementH(unsigned int depth, unsigned int row, unsigned int col) {
    Tensor3d *mat1;
    cudaMallocHost(&mat1, sizeof(Tensor3d));
    mat1->rowcount = row;
    mat1->colcount = col;
    mat1->depthCount = depth;
    mat1->elementCount = row * col * depth;
    cudaMallocHost(reinterpret_cast<void **>(&mat1->elements), row * col * depth * sizeof(float));
    return mat1;
}

Matrix::Tensor3d* Matrix::callAllocElementD(unsigned int depth, unsigned int row, unsigned int col) {
    Tensor3d *mat1;
    cudaMallocHost(&mat1, sizeof(Tensor3d));
    mat1->rowcount = row;
    mat1->colcount = col;
    mat1->depthCount = depth;
    mat1->elementCount = row * col * depth;
    cudaMalloc(reinterpret_cast<void **>(&mat1->elements), row * col * depth * sizeof(float));
    return mat1;
}

Matrix::Tensor4d* Matrix::callAllocElementH(unsigned int w, unsigned int depth, unsigned int row,
                               unsigned int col) {
    Tensor4d *mat1;
    cudaMallocHost(&mat1, sizeof(Tensor3d));
    mat1->rowcount = row;
    mat1->colcount = col;
    mat1->depthCount = depth;
    mat1->wCount = w;
    mat1->elementCount = row * col * depth * w;
    cudaMallocHost(reinterpret_cast<void **>(&mat1->elements), row * col * depth * w * sizeof(float));
    return mat1;
}

Matrix::Tensor4d* Matrix::callAllocElementD(unsigned int w, unsigned int depth, unsigned int row,
                               unsigned int col) {
    Tensor4d *mat1;
    cudaMallocHost(&mat1, sizeof(Tensor3d));
    mat1->rowcount = row;
    mat1->colcount = col;
    mat1->depthCount = depth;
    mat1->wCount = w;
    mat1->elementCount = row * col * depth * w;
    cudaMalloc(reinterpret_cast<void **>(&mat1->elements), row * col * depth * w * sizeof(float));
    return mat1;
}

void Matrix::callAllocRandom(Matrix::Matrix2d *mat1) {
    LARGE_INTEGER cpuFre;
    LARGE_INTEGER begin;

    QueryPerformanceFrequency(&cpuFre);
    QueryPerformanceCounter(&begin);
    dim3 gridSize = dim3((mat1->colcount + CUDA_BLOCK_SIZE.x - 1) / CUDA_BLOCK_SIZE.x,
                         (mat1->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    allocRandom<<<gridSize, CUDA_BLOCK_SIZE>>>((long) (&begin.QuadPart), mat1);
    cudaDeviceSynchronize();
}

void Matrix::callAllocZero(Matrix::Matrix2d *mat1) {
    cudaMemset(mat1->elements, 0.0f, sizeof(float)*mat1->colcount * mat1->rowcount);
}

void Matrix::callAllocConst(Matrix::Matrix2d *mat1, float in) {
    dim3 gridSize = dim3((mat1->colcount + CUDA_BLOCK_SIZE.x - 1) / CUDA_BLOCK_SIZE.x,
                         (mat1->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    allocConst<<<gridSize, CUDA_BLOCK_SIZE>>>(mat1, in);
    cudaDeviceSynchronize();
}

void Matrix::callAllocRandom(Matrix::Tensor *mat1) {
    LARGE_INTEGER cpuFre;
    LARGE_INTEGER begin;

    QueryPerformanceFrequency(&cpuFre);
    QueryPerformanceCounter(&begin);
    dim3 gridSize = dim3((mat1->elementCount + CUDA_BLOCK_SIZE_3D.x - 1) / CUDA_BLOCK_SIZE_3D.x);
    allocRandom<<<gridSize, CUDA_BLOCK_SIZE.x*CUDA_BLOCK_SIZE.y>>>((long) (&begin.QuadPart), mat1);
    cudaDeviceSynchronize();
}

void Matrix::callAllocZero(Matrix::Tensor *mat1) {
    cudaMemset(mat1->elements, 0.0f, sizeof(float)*mat1->elementCount);
}

void Matrix::callAllocConst(Matrix::Tensor *mat1, float in) {
    dim3 gridSize = dim3((mat1->elementCount + CUDA_BLOCK_SIZE_3D.x - 1) / CUDA_BLOCK_SIZE_3D.x);
    allocConst<<<gridSize, CUDA_BLOCK_SIZE.x*CUDA_BLOCK_SIZE.y>>>(mat1, in);
    cudaDeviceSynchronize();
}

Matrix::Matrix2d *Matrix::callCopyD2D(Matrix::Matrix2d *src, Matrix::Matrix2d *dist) {
    assert(src->rowcount * src->colcount == dist->rowcount * dist->colcount);
    cudaMemcpy(dist->elements, src->elements, sizeof(float) * src->rowcount * src->colcount, cudaMemcpyDeviceToDevice);
    return dist;
}

Matrix::Matrix2d *Matrix::callCopyD2H(Matrix::Matrix2d *src, Matrix::Matrix2d *dist) {
    assert(src->rowcount * src->colcount == dist->colcount * dist->rowcount);
    cudaMemcpy(dist->elements, src->elements, sizeof(float) * src->rowcount * src->colcount, cudaMemcpyDeviceToHost);
    return dist;
}

Matrix::Matrix2d *Matrix::callCopyH2D(Matrix::Matrix2d *src, Matrix::Matrix2d *dist) {
    assert(src->rowcount * src->colcount == dist->colcount * dist->rowcount);
    cudaMemcpy(dist->elements, src->elements, sizeof(float) * src->rowcount * src->colcount, cudaMemcpyHostToDevice);
    return dist;
}



//method callings
// ===============================================================

Matrix::Matrix2d *Matrix::callCrossPOLD(Matrix::Matrix2d *mat1, Matrix::Matrix2d *mat2, Matrix::Matrix2d *result) {
    dim3 gridSize = dim3((mat1->colcount + CUDA_BLOCK_SIZE.x - 1) / CUDA_BLOCK_SIZE.x,
                         (mat1->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    crossP<<<gridSize, CUDA_BLOCK_SIZE>>>(mat1, mat2, result);
    cudaDeviceSynchronize();
    return result;
}

Matrix::Matrix2d *Matrix::callCross(Matrix2d *mat1, Matrix2d *mat2, Matrix2d *result) {
    assert(CUDA_BLOCK_SIZE.x == CUDA_BLOCK_SIZE.y && TILE_SIZE == CUDA_BLOCK_SIZE.x);
    assert(mat1->colcount == mat2->rowcount && mat1->rowcount == result->rowcount &&
           mat2->colcount == result->colcount);
    dim3 gridSize = dim3((mat2->colcount + CUDA_BLOCK_SIZE.x - 1) / (CUDA_BLOCK_SIZE.x),
                         (mat1->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    crossTiling<<<gridSize, CUDA_BLOCK_SIZE>>>(mat1, mat2, result);
    cudaDeviceSynchronize();
    return result;
}

Matrix::Matrix2d *Matrix::callCrossCompOpt(Matrix::Matrix2d *mat1, Matrix::Matrix2d *mat2, Matrix::Matrix2d *result) {
    assert(mat1->colcount == mat2->rowcount && mat1->rowcount == result->rowcount &&
           mat2->colcount == result->colcount);
    dim3 blockSize = dim3(TILE_SIZE, VECTOR_SIZE);
    dim3 grid = dim3((mat2->colcount + (TILE_SIZE * VECTOR_SIZE) - 1) /
                     (TILE_SIZE * VECTOR_SIZE), (mat1->rowcount + TILE_SIZE - 1) / TILE_SIZE);
    crossCompOpt<<<grid, blockSize>>>(mat1, mat2, result);
    cudaDeviceSynchronize();
    return result;
}

Matrix::Matrix2d *Matrix::callCrossWMMA(Matrix::Matrix2d *mat1, Matrix::Matrix2d *mat2, Matrix::Matrix2d *result) {
    assert(mat1->rowcount % 16 == 0 && mat2->colcount % 16 == 0);
    assert(mat1->colcount % 8 == 0 && mat1->colcount == mat2->rowcount);
    assert(mat1->rowcount == result->rowcount);
    assert(mat2->colcount == result->colcount);

    dim3 blockSize = dim3(128, 4);
    dim3 gridSize = dim3(
            (mat1->colcount + (M * blockSize.x / 32 - 1)) / (M * blockSize.x / 32),
            (mat1->rowcount + N * blockSize.y - 1) / (N * blockSize.y));
    crossWMMA<<<gridSize, blockSize>>>(mat1, mat2, result);
    cudaDeviceSynchronize();
    return result;
}

Matrix::Matrix2d *
Matrix::callCrossTilingWMMA(Matrix::Matrix2d *mat1, Matrix::Matrix2d *mat2, Matrix::Matrix2d *result) {
    assert(mat1->rowcount % 16 == 0 && mat2->colcount % 16 == 0);
    assert(mat1->colcount % 8 == 0 && mat1->colcount == mat2->rowcount);
    assert(mat1->rowcount == result->rowcount);
    assert(mat2->colcount == result->colcount);

    dim3 blockSize = dim3(128, 4);
    dim3 gridSize = dim3(
            (mat1->colcount + (M * blockSize.x / 32 - 1)) / (M * blockSize.x / 32),
            (mat1->rowcount + N * blockSize.y - 1) / (N * blockSize.y));
    crossTilingWMMA<<<gridSize, blockSize>>>(mat1, mat2, result);
    cudaDeviceSynchronize();
    return result;
}

Matrix::Matrix2d *
Matrix::callCrossPrefetching(Matrix::Matrix2d *mat1, Matrix::Matrix2d *mat2, Matrix::Matrix2d *result) {
    assert(mat1->colcount == mat2->rowcount);
    assert(mat1->rowcount == result->rowcount);
    assert(mat2->colcount == result->colcount);
    dim3 blockSize = dim3(TILE_SIZE, VECTOR_SIZE);
    dim3 grid = dim3((mat2->colcount + (TILE_SIZE * VECTOR_SIZE) - 1) /
                     (TILE_SIZE * VECTOR_SIZE), (mat1->rowcount + TILE_SIZE - 1) / TILE_SIZE);
    crossPrefetching<<<grid, blockSize>>>(mat1, mat2, result);
    cudaDeviceSynchronize();
    return result;
}

Matrix::Matrix2d *
Matrix::callCrossPrefetchingA(Matrix::Matrix2d *mat1, Matrix::Matrix2d *mat2, Matrix::Matrix2d *result) {
    assert(mat1->colcount == mat2->rowcount && mat1->rowcount == result->rowcount &&
           mat2->colcount == result->colcount);
    dim3 blockSize = dim3(TILE_SIZE, VECTOR_SIZE);
    dim3 grid = dim3((mat2->colcount + (TILE_SIZE * VECTOR_SIZE) - 1) /
                     (TILE_SIZE * VECTOR_SIZE), (mat1->rowcount + TILE_SIZE - 1) / TILE_SIZE);
    crossPrefetchingA<<<grid, blockSize>>>(mat1, mat2, result);
    cudaDeviceSynchronize();
    return result;
}

Matrix::Matrix2d *Matrix::callHadamardP(Matrix::Matrix2d *mat1, Matrix::Matrix2d *mat2) {
    dim3 gridSize = dim3((mat1->colcount + CUDA_BLOCK_SIZE.x - 1) / (CUDA_BLOCK_SIZE.x),
                         (mat1->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    hadmardProduct<<<gridSize, CUDA_BLOCK_SIZE>>>(mat1, mat2);
    cudaDeviceSynchronize();
    return mat1;
}

Matrix::Matrix2d *Matrix::callConstantP(Matrix::Matrix2d *mat1, float con) {
    dim3 gridSize = dim3((mat1->colcount + CUDA_BLOCK_SIZE.x - 1) / (CUDA_BLOCK_SIZE.x),
                         (mat1->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    constantProduct<<<gridSize, CUDA_BLOCK_SIZE>>>(mat1, con);
    cudaDeviceSynchronize();
    return mat1;
}

Matrix::Matrix2d *Matrix::callAddition(Matrix::Matrix2d *mat1, Matrix::Matrix2d *mat2) {
    assert(mat1->rowcount == mat2->rowcount && mat1->colcount && mat2->rowcount);
    dim3 gridSize = dim3((mat1->colcount + CUDA_BLOCK_SIZE.x - 1) / (CUDA_BLOCK_SIZE.x),
                         (mat1->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    addition<<<gridSize, CUDA_BLOCK_SIZE>>>(mat1, mat2);
    cudaDeviceSynchronize();
    return mat1;
}

Matrix::Matrix2d *Matrix::callAddition(Matrix::Matrix2d *mat1, float con) {
    dim3 gridSize = dim3((mat1->colcount + CUDA_BLOCK_SIZE.x - 1) / (CUDA_BLOCK_SIZE.x),
                         (mat1->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    addition<<<gridSize, CUDA_BLOCK_SIZE>>>(mat1, con);
    cudaDeviceSynchronize();
    return mat1;
}

Matrix::Matrix2d *Matrix::callSubtraction(Matrix::Matrix2d *mat1, Matrix::Matrix2d *mat2) {
    assert(mat1->rowcount == mat2->rowcount && mat1->colcount && mat2->rowcount);
    dim3 gridSize = dim3((mat1->colcount + CUDA_BLOCK_SIZE.x - 1) / (CUDA_BLOCK_SIZE.x),
                         (mat1->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    subtraction<<<gridSize, CUDA_BLOCK_SIZE>>>(mat1, mat2);
    cudaDeviceSynchronize();
    return mat1;
}


Matrix::Matrix2d *Matrix::callSubtraction(Matrix::Matrix2d *mat1, float con) {
    dim3 gridSize = dim3((mat1->colcount + CUDA_BLOCK_SIZE.x - 1) / (CUDA_BLOCK_SIZE.x),
                         (mat1->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    subtraction<<<gridSize, CUDA_BLOCK_SIZE>>>(mat1, con);
    cudaDeviceSynchronize();
    return mat1;
}


Matrix::Matrix2d *Matrix::callPower(Matrix::Matrix2d *mat1, float con) {
    dim3 gridSize = dim3((mat1->colcount + CUDA_BLOCK_SIZE.x - 1) / (CUDA_BLOCK_SIZE.x),
                         (mat1->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    power<<<gridSize, CUDA_BLOCK_SIZE>>>(mat1, con);
    cudaDeviceSynchronize();
    return mat1;
}

Matrix::Matrix2d *Matrix::callTranspose(Matrix::Matrix2d *mat1, Matrix::Matrix2d *result) {
    assert(mat1->rowcount == result->colcount && mat1->colcount && result->rowcount);
    dim3 gridSize = dim3((mat1->colcount + CUDA_BLOCK_SIZE.x - 1) / (CUDA_BLOCK_SIZE.x),
                         (mat1->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    transposeOperation<<<gridSize, CUDA_BLOCK_SIZE>>>(mat1, result);
    cudaDeviceSynchronize();
    return result;
}

//debug tools
void Matrix::inspect(Matrix2d *mat1) {
    auto *debug = callAllocElementH(mat1->rowcount, mat1->colcount);
    cudaMemcpy(debug->elements, mat1->elements, sizeof(float) * debug->colcount * debug->rowcount,
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < debug->rowcount; i++) {
        for (int j = 0; j < debug->colcount; j++) {
            std::cout << (*(debug->elements + i * debug->colcount + j)) << " ";
        }
        std::cout << std::endl;
    }
    cudaFree(debug->elements);
    cudaFree(debug);
}

void Matrix::inspect(Tensor3d *mat1) {
    Matrix::Tensor3d *debug;
    debug = callAllocElementH(mat1->depthCount, mat1->rowcount, mat1->colcount);
    cudaMemcpy(debug->elements, mat1->elements, sizeof(float) * debug->depthCount * debug->colcount * debug->rowcount,
               cudaMemcpyDeviceToHost);
    for(int d = 0; d < debug->depthCount; d++) {
        for (int i = 0; i < debug->rowcount; i++) {
            for (int j = 0; j < debug->colcount; j++) {
                std::cout << (*(debug->elements + d*debug->rowcount*debug->colcount + i * debug->colcount + j)) << " ";
            }
            std::cout << std::endl;
        }
        std::cout <<"\n"<< std::endl;
    }
    cudaFree(debug->elements);
    cudaFree(debug);
}


void reduce(float *input, float *output, unsigned int size) {
    unsigned int procSize = size;
    unsigned int bSize = CUDA_BLOCK_SIZE.x * CUDA_BLOCK_SIZE.y;
    float *proc;
    cudaMalloc((void **) &proc, sizeof(float) * size);
    cudaMemcpy(proc, input, sizeof(float) * size, cudaMemcpyDeviceToDevice);
    while (procSize / WARP_SIZE > 0) {
        reduction<<<procSize / bSize + 1, bSize>>>(procSize, proc, proc);
        procSize = procSize % WARP_SIZE ? procSize / WARP_SIZE + 1 : procSize / WARP_SIZE;
        cudaDeviceSynchronize();
    }
    reduction<<<1, bSize>>>(procSize, proc, proc);
    cudaDeviceSynchronize();
    cudaMemcpy(output, proc, sizeof(float), cudaMemcpyDeviceToHost);
}

void Matrix::callSum(Matrix::Matrix2d *mat1, float *sumBuffer) {
    reduce(mat1->elements, sumBuffer, mat1->colcount * mat1->rowcount);
};

//batch operation
Matrix::Matrix2d *Matrix::callInsertCol(Matrix::Matrix2d *mat1, Matrix::Matrix2d *column, unsigned int colIndex) {
    assert(colIndex < mat1->colcount && column->rowcount == mat1->rowcount);
    dim3 grid = dim3((column->rowcount + (CUDA_BLOCK_SIZE.x - 1)) / CUDA_BLOCK_SIZE.x, 1);
    insertColumn<<<grid, CUDA_BLOCK_SIZE>>>(mat1, column, colIndex);
    cudaDeviceSynchronize();
    return mat1;
}


//operators
Matrix::Matrix2d *Matrix::Matrix2d::operator+(Matrix::Matrix2d *mat2) {
    return callAddition(this, mat2);
}

Matrix::Matrix2d *Matrix::Matrix2d::operator+(float con) {
    return callAddition(this, con);
}

Matrix::Matrix2d *Matrix::Matrix2d::operator-(Matrix::Matrix2d *mat2) {
    return callSubtraction(this, mat2);
}

Matrix::Matrix2d *Matrix::Matrix2d::operator-(float con) {
    return callSubtraction(this, con);
}

Matrix::Matrix2d *Matrix::Matrix2d::operator^(float con) {
    return callPower(this, con);
}

Matrix::Matrix2d *Matrix::Matrix2d::operator*(float con) {
    return callConstantP(this, con);
}

Matrix::Matrix2d *Matrix::Matrix2d::operator*(Matrix::Matrix2d *mat2) {
    return callHadamardP(this, mat2);
}

Matrix::Matrix2d *Matrix::Matrix2d::operator+=(Matrix::Matrix2d *mat2) {
    return callAddition(this, mat2);
}

Matrix::Matrix2d *Matrix::Matrix2d::operator+=(float con) {
    return callAddition(this, con);
}

void Matrix::Matrix2d::setH2D(unsigned int row, unsigned int col, float value) {
    assert(row < this->rowcount && col < this->colcount);
    cudaMemcpy(this->elements + row*colcount + col, &value, sizeof(float), cudaMemcpyHostToDevice);
}

__host__ void Matrix::Matrix2d::setH2D(unsigned int offset, float value) {
    assert(offset < this->rowcount * this->colcount);
    cudaMemcpy(this->elements + offset, &value, sizeof(float), cudaMemcpyHostToDevice);
}

__host__ float Matrix::Matrix2d::getH2D(unsigned int row, unsigned int col) const {
    float output=0;
    cudaMemcpy(&output, this->elements + row * this->colcount + col, sizeof(float), cudaMemcpyDeviceToHost);
    return output;
}

__host__ string Matrix::Matrix2d::toString() const {
    return "(" + to_string(rowcount) + "," + to_string(colcount) + ")";
}

__host__ string Matrix::Tensor4d::toString() const {
    return "(" + to_string(wCount) + "," + to_string(depthCount) + "," + to_string(rowcount) + "," + to_string(colcount) + ")";
}

dim4::dim4(unsigned int x, unsigned int y, unsigned int z, unsigned int w) : x(x), y(y), z(z), w(w) {}
