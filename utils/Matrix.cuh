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
#include <string>

static const dim3 CUDA_BLOCK_SIZE = dim3(16, 16);
static const dim3 CUDA_BLOCK_SIZE_3D = dim3(16,16,4);
static const unsigned int WARP_SIZE = 32;
static const unsigned int TILE_SIZE = 16;
static const unsigned int VECTOR_SIZE = 4;
using namespace std;

struct dim4{
    unsigned int x;
    unsigned int y;
    unsigned int z;
    unsigned int w;

    dim4(unsigned int x, unsigned int y, unsigned int z, unsigned int w);
};

//Mo stands for Matrix Operations
class Matrix {
public:
    struct Matrix2d{
        unsigned int rowcount;
        unsigned int colcount;
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
        __device__ float get(unsigned int row, unsigned int col) const;
        __device__ void set(unsigned int row, unsigned int col, float value) const;
        __device__ void add(unsigned int row, unsigned int col, float value) const;
        __device__ void atomAdd(unsigned int row, unsigned int col, float value) const;

        __host__ void setH2D(unsigned int row, unsigned int col, float value);
        __host__ void setH2D(unsigned int offset, float value);
        __host__ float getH2D(unsigned int row, unsigned int col) const;
        __host__ string toString() const;

        __host__ void index(unsigned int row, unsigned int col, float* elements);
        __host__ void index(unsigned int row, unsigned int col);
    };

    struct Tensor{
    public:
        unsigned int elementCount;
        float* elements;
        Tensor* operator+(Tensor* mat2);
        Tensor* operator-(Tensor* mat2);
        Tensor* operator*(Tensor* mat2);
        Tensor* operator*(float val);
    };

    struct Tensor3d : public Tensor{
        unsigned int depthCount;
        unsigned int rowcount;
        unsigned int colcount;

        __device__ float get(unsigned int depth, unsigned int row, unsigned int col) const;
        __device__ float get(unsigned int depth, unsigned int offset) const;
        __device__ void set(unsigned int depth, unsigned int row, unsigned int col, float value) const;
        __device__ void set(unsigned int depth, unsigned int offset, float value) const;
        __device__ void add(unsigned int depth, unsigned int row, unsigned int col, float value) const;
        __device__ void atomAdd(unsigned int depth, unsigned int row, unsigned int col, float value) const;

        __host__ void index(unsigned int depth, unsigned int row, unsigned int col, float* elements);
        __host__ void index(unsigned int depth, unsigned int row, unsigned int col);

        __host__ void extract2d(unsigned int depth, Matrix2d* mat) const;
        __host__ void emplace2d(unsigned int depth, Matrix2d* mat) const;
        __host__ string toString() const;
    };

    struct Tensor4d : public Tensor{
        unsigned int wCount;
        unsigned int depthCount;
        unsigned int rowcount;
        unsigned int colcount;

        __device__ float get(unsigned int w, unsigned int depth, unsigned int row, unsigned int col) const;
        __device__ void set(unsigned int w, unsigned int depth, unsigned int row, unsigned int col, float value) const;

        __host__ string toString() const;
    };

    //faster calculation methods
    static __device__ float fasterSqrt(float in);

    //debug tools
    static void inspect(Matrix2d *mat1);
    static void inspect(Tensor3d *mat1);

    //memory control and element allocation
    static Matrix2d* callAllocElementH(unsigned int row, unsigned int col);
    static Matrix2d* callAllocElementD(unsigned int row, unsigned int col);
    static Tensor3d* callAllocElementH(unsigned int depth, unsigned int row, unsigned int col);
    static Tensor3d* callAllocElementD(unsigned int depth, unsigned int row, unsigned int col) ;
    static Tensor4d* callAllocElementH(unsigned int w, unsigned int depth, unsigned int row, unsigned int col);
    static Tensor4d* callAllocElementD(unsigned int w, unsigned int depth, unsigned int row, unsigned int col);

    static void callAllocRandom(Matrix2d *mat1);
    static void callAllocZero(Matrix2d *mat1);
    static void callAllocConst(Matrix2d *mat1, float in);

    static void callAllocRandom(Tensor *mat1);
    static void callAllocZero(Tensor *mat1);
    static void callAllocConst(Tensor *mat1, float in);

    static Matrix2d* callCopyD2D(Matrix2d *src, Matrix2d *dist);
    static Matrix2d* callCopyD2H(Matrix2d *src, Matrix2d* dist);
    static Matrix2d* callCopyH2D(Matrix2d *src, Matrix2d* dist);

    //method callings
    static Matrix2d* callCross(Matrix2d* mat1, Matrix2d* mat2, Matrix2d* result);
    static Matrix2d* callCrossWMMA(Matrix2d *mat1, Matrix2d *mat2, Matrix2d *result);
    static Matrix2d* callCrossTilingWMMA(Matrix2d *mat1, Matrix2d *mat2, Matrix2d *result);
    static Matrix2d* callCrossPrefetching(Matrix2d *mat1, Matrix2d *mat2, Matrix2d *result);  //A*B
    static Matrix2d* callCrossPrefetchingA(Matrix2d *mat1, Matrix2d *mat2, Matrix2d *result);  //A*B + C
    static Matrix2d* callCrossCompOpt(Matrix2d* mat1, Matrix2d* mat2, Matrix2d* result);
    static Matrix2d* callCrossPOLD(Matrix2d* mat1, Matrix2d* mat2, Matrix2d* result);
    static Matrix2d* callTranspose(Matrix2d* mat1, Matrix2d* result);
    static void callSum(Matrix2d *mat1, float* sumBuffer);

    //edit element in batches
    static Matrix2d* callInsertCol(Matrix2d* mat1, Matrix2d* column, unsigned int colIndex);

    //operator methods and normal operations
    static Matrix2d* callHadamardP(Matrix2d* mat1, Matrix2d* mat2);
    static Matrix2d* callConstantP(Matrix2d* mat1, float con);
    static Matrix2d* callAddition(Matrix2d* mat1, Matrix2d* mat2);
    static Matrix2d* callAddition(Matrix2d* mat1, float con);
    static Matrix2d* callSubtraction(Matrix2d* mat1, Matrix2d* mat2);
    static Matrix2d* callSubtraction(Matrix2d* mat1, float con);
    static Matrix2d* callPower(Matrix2d* mat1, float con);
};

//method that does not need the class name (for clarity)
static Matrix::Matrix2d* cross(Matrix::Matrix2d* mat1, Matrix::Matrix2d* mat2, Matrix::Matrix2d* result) {
    return Matrix::callCrossPrefetching(mat1, mat2, result);
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

static Matrix::Matrix2d* insertCol(Matrix::Matrix2d* mat1, Matrix::Matrix2d* column, unsigned int colIndex){
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
    auto* host = Matrix::callAllocElementH(mat1->rowcount, mat1->colcount);
    copyH2D(mat1,host);
    for (unsigned int i=0; i<mat1->rowcount* mat1->colcount; i++){
        sum+=host->elements[i];
    }
    cudaFreeHost(host->elements);
    cudaFreeHost(host);
    return sum;
}

static Matrix::Matrix2d* flatten(Matrix::Matrix2d* in){
    in->rowcount*= in->colcount;
    in->colcount = 1;
    return in;
}

#endif //CUDANNGEN2_MATRIX_CUH
