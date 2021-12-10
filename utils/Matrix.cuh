//
// Created by DanielSun on 9/23/2021.
//

#ifndef CUDANNGEN2_MATRIX_CUH
#define CUDANNGEN2_MATRIX_CUH

#include<cuda.h>
#include<cuda_runtime.h>
#include <curand_kernel.h>
#include <ctime>
#include <mma.h>
#include <cassert>

static const dim3 CUDA_BLOCK_SIZE = dim3(16, 16);
static const int WARP_SIZE = 32;
static const int TILE_SIZE = 16;
static const int VECTOR_SIZE = 4;

//Mo stands for Matrix Operations
class Matrix {
public:
    struct Matrix2d{
        int rowcount;
        int colcount;
        float* elements;

        //operator definitions
        Matrix2d* operator+(Matrix2d* mat2);
        Matrix2d* operator+=(Matrix2d* mat2);
        Matrix2d* operator+=(float con);
        Matrix2d* operator+(float con);
        Matrix2d* operator-(Matrix2d* mat2);
        Matrix2d* operator-(float con);
        Matrix2d* operator^(float con);
        Matrix2d* operator*(Matrix2d* mat2);  //hadmard product, not cross product
        Matrix2d* operator*(float con);

        //get the element at the particular location
        __device__ float get(int row, int col) const  ;
        __device__ void set(int row, int col, float value) const ;
        __device__ void add(int row, int col, float value) const ;

        __host__ void setH2D(int row, int col, float value);
        __host__ void setH2D(int offset, float value);
        __host__ float getH2D(int row, int col) const;
    };

    //faster calculation methods
    static __device__ float fasterSqrt(float in);

    //debug tools
    static void inspect(Matrix2d *mat1);

    //memory control and element allocation
    static void callAllocElementH(Matrix2d *mat1, int row, int col);
    static void callAllocElementD(Matrix2d *mat1, int row, int col);
    static void callAllocRandom(Matrix2d *mat1);
    static void callAllocZero(Matrix2d *mat1);
    static void callAllocConst(Matrix2d *mat1, float in);
    static Matrix2d* callCopyD2D(Matrix2d *src, Matrix2d *dist);
    static Matrix2d* callCopyD2H(Matrix2d *src, Matrix2d* dist);
    static Matrix2d* callCopyH2D(Matrix2d *src, Matrix2d* dist);

    //method callings
    static Matrix2d* callCross(Matrix2d* mat1, Matrix2d* mat2, Matrix2d* result);
    static Matrix2d* callCrossPrefetching(Matrix2d *mat1, Matrix2d *mat2, Matrix2d *result);
    static Matrix2d* callCrossPrefetchingA(Matrix2d *mat1, Matrix2d *mat2, Matrix2d *result);
    static Matrix2d* callCrossCompOpt(Matrix2d* mat1, Matrix2d* mat2, Matrix2d* result);
    static Matrix2d* callCrossPOLD(Matrix2d* mat1, Matrix2d* mat2, Matrix2d* result);
    static Matrix2d* callTranspose(Matrix2d* mat1, Matrix2d* result);
    static void callSum(Matrix2d *mat1, float* sumBuffer);

    //edit element in batches
    static Matrix2d* callInsertCol(Matrix2d* mat1, Matrix2d* column, int colIndex);

    //operator methods and normal operations
    static Matrix2d* callHadmardP(Matrix2d* mat1, Matrix2d* mat2);
    static Matrix2d* callConstantP(Matrix2d* mat1, float con);
    static Matrix2d* callAddition(Matrix2d* mat1, Matrix2d* mat2);
    static Matrix2d* callAddition(Matrix2d* mat1, float con);
    static Matrix2d* callSubtraction(Matrix2d* mat1, Matrix2d* mat2);
    static Matrix2d* callSubtraction(Matrix2d* mat1, float con);
    static Matrix2d* callPower(Matrix2d* mat1, float con);


};

//method that does not need the class name (for clarity)
static Matrix::Matrix2d* cross(Matrix::Matrix2d* mat1, Matrix::Matrix2d* mat2, Matrix::Matrix2d* result){
    return Matrix::callCrossPrefetching(mat1,mat2,result);
}

static Matrix::Matrix2d* crossA(Matrix::Matrix2d* mat1, Matrix::Matrix2d* mat2, Matrix::Matrix2d* result){
    return Matrix::callCrossPrefetchingA(mat1,mat2,result);
}

static Matrix::Matrix2d* transpose(Matrix::Matrix2d* mat1, Matrix::Matrix2d* result){
    return Matrix::callTranspose(mat1, result);
}

static Matrix::Matrix2d* copyD2D(Matrix::Matrix2d* src, Matrix::Matrix2d* dist){
    return Matrix::callCopyD2D(src, dist);
}

static Matrix::Matrix2d* copyD2H(Matrix::Matrix2d* src, Matrix::Matrix2d* dist){
    return Matrix::callCopyD2H(src,dist);
}

static Matrix::Matrix2d* copyH2D(Matrix::Matrix2d* src, Matrix::Matrix2d* dist){
    return Matrix::callCopyH2D(src,dist);
}

static Matrix::Matrix2d* insertCol(Matrix::Matrix2d* mat1, Matrix::Matrix2d* column, int colIndex){
    return Matrix::callInsertCol(mat1, column, colIndex);
}

static float sumH(Matrix::Matrix2d* mat1){
    float* onDevice, *onHost;
    cudaMalloc((void**)&onDevice, sizeof(float));
    cudaMallocHost((void**)&onHost, sizeof(float));
    assert(onDevice!=nullptr && onHost != nullptr);
    Matrix::callSum(mat1, onDevice);
    cudaMemcpy(onHost, onDevice, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(onDevice);
    float out = *onHost;
    cudaFree(onHost);
    return out;
}

static void sum(Matrix::Matrix2d* mat1, float* buffer){
   Matrix::callSum(mat1, buffer);
}

static float sumC(Matrix::Matrix2d* mat1){
    float sum = 0;
    Matrix::Matrix2d* host;
    cudaMallocHost((void**)&host, sizeof(Matrix::Matrix2d));
    Matrix::callAllocElementH(host, mat1->rowcount, mat1->colcount);
    copyH2D(mat1,host);
    for (int i=0; i<mat1->rowcount* mat1->colcount; i++){
        sum+=host->elements[i];
    }
    cudaFreeHost(host->elements);
    cudaFreeHost(host);
    return sum;
}

static Matrix::Matrix2d* flattern(Matrix::Matrix2d* in){
    in->rowcount*= in->colcount;
    in->colcount = 1;
    return in;
}

#endif //CUDANNGEN2_MATRIX_CUH
