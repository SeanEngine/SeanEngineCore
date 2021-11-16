//
// Created by DanielSun on 9/23/2021.
//

#ifndef CUDANNGEN2_MATRIX_CUH
#define CUDANNGEN2_MATRIX_CUH

#include<cuda.h>
#include<cuda_runtime.h>
#include <curand_kernel.h>
#include <ctime>


static const dim3 CUDA_BLOCK_SIZE = dim3(16, 16);
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
    static Matrix2d* callCopyD2D(Matrix2d *src, Matrix2d *dist);

    //method callings
    static Matrix2d* callCross(Matrix2d* mat1, Matrix2d* mat2, Matrix2d* result);
    static Matrix2d* callCrossPrefetching(Matrix2d *mat1, Matrix2d *mat2, Matrix2d *result);
    static Matrix2d* callCrossPrefetchingA(Matrix2d *mat1, Matrix2d *mat2, Matrix2d *result);
    static Matrix2d* callCrossCompOpt(Matrix2d* mat1, Matrix2d* mat2, Matrix2d* result);
    static Matrix2d* callCrossPOLD(Matrix2d* mat1, Matrix2d* mat2, Matrix2d* result);
    static Matrix2d* callTranspose(Matrix2d* mat1, Matrix2d* result);
    static void callSum(Matrix2d *mat1, float* sumBuffer);

    //operator methods and normal operations
    static Matrix2d* callHadmardP(Matrix2d* mat1, Matrix2d* mat2);
    static Matrix2d* callConstantP(Matrix2d* mat1, float con);
    static Matrix2d* callAddition(Matrix2d* mat1, Matrix2d* mat2);
    static Matrix2d* callAddition(Matrix2d* mat1, float con);
    static Matrix2d* callSubtraction(Matrix2d* mat1, Matrix2d* mat2);
    static Matrix2d* callSubtraction(Matrix2d* mat1, float con);
    static Matrix2d* callPower(Matrix2d* mat1, float con);

    //activation methods
    static Matrix2d* callActivationSigmoid(Matrix2d* mat1);
    static Matrix2d* callActivationSigmoid(Matrix2d *mat1, Matrix2d *result);
    static Matrix2d* callDerivativeSigmoid(Matrix2d *mat1, Matrix2d *result);
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

static Matrix::Matrix2d* sigmoid(Matrix::Matrix2d* mat1, Matrix::Matrix2d* result){
    return Matrix::callActivationSigmoid(mat1, result);
}

static Matrix::Matrix2d* sigmoidD(Matrix::Matrix2d* mat1, Matrix::Matrix2d* result){
    return Matrix::callDerivativeSigmoid(mat1, result);
}

static Matrix::Matrix2d* copyD2D(Matrix::Matrix2d* src, Matrix::Matrix2d* dist){
    return Matrix::callCopyD2D(src, dist);
}

static void sum(Matrix::Matrix2d* mat1, float* result){
    Matrix::callSum(mat1, result);
}

#endif //CUDANNGEN2_MATRIX_CUH
