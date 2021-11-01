//
// Created by DanielSun on 9/23/2021.
//

#ifndef CUDANNGEN2_MATRIX_CUH
#define CUDANNGEN2_MATRIX_CUH

#include<cuda.h>
#include<cuda_runtime.h>
#include <curand_kernel.h>
#include <ctime>


static const dim3 CUDA_BLOCK_SIZE = dim3(16,16);
static const int TILE_SIZE = 16;
static const int VECTOR_SIZE = 4;

//Mo stands for Matrix Operations
class Matrix {
public:
    struct Matrix2d{
        int rowcount;
        int colcount;
        float* elements;

        Matrix2d* operator+(Matrix2d* mat2);
        Matrix2d* operator+(float con);
        Matrix2d* operator-(Matrix2d* mat2);
        Matrix2d* operator-(float con);
        Matrix2d* operator^(float con);
        Matrix2d* operator*(float con);

        //get the element at the particular location
        __device__ float get(int row, int col) const  ;
        __device__ void set(int row, int col, float value) const ;
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

    //method callings
    static Matrix2d* callCrossP(Matrix2d* mat1, Matrix2d* mat2, Matrix2d* result);
    static Matrix2d* callCrossPrefetching(Matrix2d *mat1, Matrix2d *mat2, Matrix2d *result);
    static Matrix2d* callCrossCompOpt(Matrix2d* mat1, Matrix2d* mat2, Matrix2d* result);
    static Matrix2d* callCrossPOlD(Matrix2d* mat1, Matrix2d* mat2, Matrix2d* result);

    //operator methods and normal operations
    static Matrix2d* callConstantP(Matrix2d* mat1, float con);
    static Matrix2d* callAddition(Matrix2d* mat1, Matrix2d* mat2);
    static Matrix2d* callAddition(Matrix2d* mat1, float con);
    static Matrix2d* callSubtraction(Matrix2d* mat1, Matrix2d* mat2);
    static Matrix2d* callSubtraction(Matrix2d* mat1, float con);
    static Matrix2d* callPower(Matrix2d* mat1, float con);
};

//method that does not need the class name (for clarity)
static Matrix::Matrix2d* cross(Matrix::Matrix2d* mat1, Matrix::Matrix2d* mat2, Matrix::Matrix2d* result){
    return Matrix::callCrossP(mat1,mat2,result);
}

#endif //CUDANNGEN2_MATRIX_CUH
