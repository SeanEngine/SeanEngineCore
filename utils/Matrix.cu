//
// Created by DanielSun on 9/23/2021.
//

#include "Matrix.cuh"
#include <string>
#include <iostream>
#include <cassert>

using namespace std;

//device methods
__device__ float Matrix::Matrix2d::get(int row, int col) const   {
    return (row < rowcount && col < colcount) ? this->elements[row * this->colcount + col] : 0.0f;
}

__device__ void Matrix::Matrix2d::set(int row, int col, float value) const   {
    if(row < rowcount && col < colcount)
    this->elements[row * this->colcount + col] = value;
}

__device__ float Matrix::fasterSqrt(float in) {
    float half = 0.5f*in;
    int i = *(int*)&in;
    i = 0x5f375a86-(i>>1);
    in = *(float*)&i;
    in = in*(1.5f-half*in*in);
    return in;
}

//random number fill (0-1)
__global__ void allocRandom(long seed, Matrix::Matrix2d *mat1){
    curandStateXORWOW_t state;
    int row = static_cast<int>(threadIdx.y + blockIdx.y * blockDim.y);
    int col =  static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
    curand_init((row+1)*(col+1)*seed,0,0,&state);
    mat1->set(row, col, static_cast<float>((curand_uniform(&state) * 2.0F) - 1.0F));
}

//zero fill
__global__ void allocZero(Matrix::Matrix2d *mat1) {
    int row =  static_cast<int>(threadIdx.y + blockIdx.y * blockDim.y);
    int col =  static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
    mat1->set(row, col, 0.0F);
}

__global__ void crossP(Matrix::Matrix2d* mat1, Matrix::Matrix2d* mat2, Matrix::Matrix2d* result){
    float currentValue = 0.0;
    int row =  static_cast<int>(threadIdx.y + blockIdx.y * blockDim.y);
    int col =  static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);

    for (int i = 0; i < mat1->colcount; ++i) {
        currentValue += mat1->get(row, i) *
                        mat2->get(i, col);
    }

    result->set(row, col, currentValue);
}

//constants : TILE_SIZE = blockSize.x = blockSize.y,
__global__ void crossTiling(Matrix::Matrix2d* mat1, Matrix::Matrix2d* mat2, Matrix::Matrix2d* result){

    __shared__ float mat1_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float mat2_tile[TILE_SIZE][TILE_SIZE];
    float resultOutput = 0;   // result of C in register

    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    //#pragma unroll
    for (int tileId = 0; tileId < (mat1->colcount + TILE_SIZE-1)/TILE_SIZE; tileId++ ){

        //load shared memory
        mat1_tile[threadIdx.y][threadIdx.x] = mat1->get(row, threadIdx.x + tileId * TILE_SIZE);
        mat2_tile[threadIdx.y][threadIdx.x] = mat2->get(threadIdx.y + tileId * TILE_SIZE, col);
        __syncthreads();

        #pragma unroll
        for(int mulIndex = 0; mulIndex < TILE_SIZE; mulIndex++){
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
 */

__global__ void crossCompOpt(Matrix::Matrix2d* mat1, Matrix::Matrix2d* mat2, Matrix::Matrix2d* result){

    __shared__ float mat1_tile[TILE_SIZE][TILE_SIZE];
    float mat2Value = 0.0f;
    float resultBuffer[TILE_SIZE] = {0};

    int resultCol = VECTOR_SIZE*TILE_SIZE*blockIdx.x + threadIdx.y * TILE_SIZE + threadIdx.x;

    for (int tileId = 0; tileId < (mat1->colcount + TILE_SIZE-1)/TILE_SIZE; tileId++ ){
        //allocate elements
        for (int i = 0; i < TILE_SIZE / VECTOR_SIZE; i++) {
            mat1_tile[threadIdx.x][i*VECTOR_SIZE + threadIdx.y]= mat1->
                    get(blockIdx.y * TILE_SIZE + i*VECTOR_SIZE + threadIdx.y,tileId * TILE_SIZE + threadIdx.x);
        }
        __syncthreads();

        for (int row = 0; row < TILE_SIZE; row++){
            //pick a value of mat2 and put it into the registers
            mat2Value = mat2->get(tileId * TILE_SIZE + row,resultCol);
            for (int bufId = 0; bufId < TILE_SIZE; bufId++){
                resultBuffer[bufId] += mat1_tile[row][bufId] * mat2Value;
            }
        }
        __syncthreads();
    }
    int resultRow0 = blockIdx.y * TILE_SIZE;
    for (int bufId = 0; bufId <TILE_SIZE; bufId++){
        result->set(resultRow0 + bufId, resultCol, resultBuffer[bufId]);
    }
}

__global__ void prefetchingCrossP(Matrix::Matrix2d* mat1, Matrix::Matrix2d* mat2, Matrix::Matrix2d* result){

}

//memory Control:
void Matrix::callAllocElementH(Matrix::Matrix2d *mat1, int row, int col) {
    mat1->rowcount = row;
    mat1->colcount = col;
    (void)cudaMallocHost(reinterpret_cast<void**>(&mat1->elements),row*col*sizeof(float));
}

void Matrix::callAllocElementD(Matrix::Matrix2d *mat1, int row, int col) {
    mat1->rowcount = row;
    mat1->colcount = col;
    cudaMalloc(reinterpret_cast<void**>(&mat1->elements),row*col*sizeof(float));
}


void Matrix::callAllocRandom(Matrix::Matrix2d *mat1) {
    dim3 gridSize = dim3((mat1->colcount + CUDA_BLOCK_SIZE.x-1) / CUDA_BLOCK_SIZE.x,
                         (mat1->rowcount + CUDA_BLOCK_SIZE.y-1) / CUDA_BLOCK_SIZE.y);
    allocRandom<<<gridSize, CUDA_BLOCK_SIZE>>>(time(nullptr),mat1);
    (void) cudaDeviceSynchronize();
}

void Matrix::callAllocZero(Matrix::Matrix2d *mat1) {
    dim3 gridSize = dim3((mat1->colcount + CUDA_BLOCK_SIZE.x - 1) / CUDA_BLOCK_SIZE.x,
                         (mat1->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    allocZero<<<gridSize, CUDA_BLOCK_SIZE>>>(mat1);
    (void) cudaDeviceSynchronize();
}

Matrix::Matrix2d *Matrix::callCrossPOlD(Matrix::Matrix2d *mat1, Matrix::Matrix2d *mat2, Matrix::Matrix2d *result) {
    dim3 gridSize = dim3((mat1->colcount + CUDA_BLOCK_SIZE.x - 1) / CUDA_BLOCK_SIZE.x,
                         (mat1->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    (void) crossP<<<gridSize, CUDA_BLOCK_SIZE>>>(mat1, mat2, result);
    (void) cudaDeviceSynchronize();
    return result;
}

//method callings
Matrix::Matrix2d* Matrix::callCrossP(Matrix2d* mat1, Matrix2d* mat2, Matrix2d* result) {
    assert(CUDA_BLOCK_SIZE.x == CUDA_BLOCK_SIZE.y && TILE_SIZE == CUDA_BLOCK_SIZE.x);
    assert(mat1->colcount == mat2->rowcount && mat1->rowcount == result->rowcount && mat2->colcount == result->colcount);
    dim3 gridSize = dim3((mat2->colcount + CUDA_BLOCK_SIZE.x-1) / (CUDA_BLOCK_SIZE.x),
                         (mat1->rowcount + CUDA_BLOCK_SIZE.y-1) / CUDA_BLOCK_SIZE.y);
    (void) crossTiling<<<gridSize, CUDA_BLOCK_SIZE>>>(mat1, mat2, result);
    (void) cudaDeviceSynchronize();
    return result;
}

Matrix::Matrix2d *Matrix::callCrossCompOpt(Matrix::Matrix2d *mat1, Matrix::Matrix2d *mat2, Matrix::Matrix2d *result) {
    assert(mat1->colcount == mat2->rowcount && mat1->rowcount == result->rowcount && mat2->colcount == result->colcount);
    dim3 blockSize = dim3(TILE_SIZE, VECTOR_SIZE);
    dim3 grid = dim3((mat2->colcount + (TILE_SIZE * VECTOR_SIZE)-1) /
            (TILE_SIZE * VECTOR_SIZE), (mat1->rowcount + TILE_SIZE -1) / TILE_SIZE);
    (void) crossCompOpt<<<grid, blockSize>>>(mat1, mat2, result);
    (void) cudaDeviceSynchronize();
    return result;
}

Matrix::Matrix2d *Matrix::callConstantP(Matrix::Matrix2d *mat1, float con) {
    return nullptr;
}

Matrix::Matrix2d *Matrix::callAddition(Matrix::Matrix2d *mat1, Matrix::Matrix2d *mat2) {
    return nullptr;
}

Matrix::Matrix2d *Matrix::callAddition(Matrix::Matrix2d *mat1, float con) {
    return nullptr;
}

Matrix::Matrix2d *Matrix::callSubtraction(Matrix::Matrix2d *mat1, Matrix::Matrix2d *mat2) {
    return nullptr;
}

Matrix::Matrix2d *Matrix::callSubtraction(Matrix::Matrix2d *mat1, float con) {
    return nullptr;
}

Matrix::Matrix2d *Matrix::callPower(Matrix::Matrix2d *mat1, Matrix::Matrix2d *mat2) {
    return nullptr;
}

Matrix::Matrix2d *Matrix::callPower(Matrix::Matrix2d *mat1, float con) {
    return nullptr;
}

void Matrix::inspect(Matrix2d *mat1) {
    Matrix2d* debug;
    cudaMallocHost(reinterpret_cast<void**>(&debug), mat1->colcount * mat1->rowcount * sizeof(float));
    callAllocElementH(debug, mat1->rowcount, mat1->colcount);
    cudaMemcpy(debug->elements,mat1->elements,sizeof(float)*debug->colcount*debug->rowcount,cudaMemcpyDeviceToHost);
    for (int i = 0; i< debug -> rowcount; i++) {
        for (int j = 0; j < debug->colcount; j++) {
            std::cout<<*(debug->elements + i*debug->colcount + j)<<" ";
        }
        std::cout<<std::endl;
    }
    cudaFree(debug->elements);
    cudaFree(debug);
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
    return callSubtraction(this,con);
}

Matrix::Matrix2d *Matrix::Matrix2d::operator^(Matrix::Matrix2d *mat2) {
    return callPower(this, mat2);
}

Matrix::Matrix2d *Matrix::Matrix2d::operator^(float con) {
    return callPower(this, con);
}

Matrix::Matrix2d *Matrix::Matrix2d::operator*(float con) {
    return callConstantP(this, con);
}


