#include <cstdio>
#include "utils/Matrix.cuh"
#include <windows.h>
#include <string>
#include <iostream>
#include "io/Reader.cuh"
#include "cublas_v2.h"
#include "execution/ThreadControls.h"
#include "models/DenseMLP.cuh"
#include "layers/DenseLayer.cuh"

#pragma comment(lib, "cublas.lib")

using namespace std;
using namespace nvcuda;

int main(int argc, char **argv) {;
/*
    int size = 64;
    Matrix::Matrix2d*A, *B, *C;
    cudaMallocHost(&A, sizeof(Matrix::Matrix2d));
    cudaMallocHost(&B, sizeof(Matrix::Matrix2d));
    cudaMallocHost(&C, sizeof(Matrix::Matrix2d));
    Matrix::callAllocElementD(A, 64, 64);
    Matrix::callAllocElementD(B, 64, 64);
    Matrix::callAllocElementD(C, 64, 64);
    Matrix::callAllocConst(A, 0.2f);
    Matrix::callAllocConst(B, 0.2f);

    LARGE_INTEGER cpuFre;
    LARGE_INTEGER begin;
    LARGE_INTEGER end;

    QueryPerformanceFrequency(&cpuFre);
    QueryPerformanceCounter(&begin);

    Matrix::callCrossTilingWMMA(A,B,C);

    QueryPerformanceCounter(&end);
    cout<<"TensorCore Tiling GEMM : "<<(end.QuadPart - begin.QuadPart)/1e7<<endl;
    cout<<endl;
    Matrix::inspect(C);
    QueryPerformanceCounter(&begin);
    cross(A,B,C);
    QueryPerformanceCounter(&end);
    Matrix::inspect(C);
    cout<<"Prefetching : "<<(end.QuadPart - begin.QuadPart)/1e7<<endl;
*/
    auto *model = new DenseMLP();
    model->registerModel();
    model->loadModel();
    model->loadDataSet();
    model->loadData();
    //Matrix::inspect(((DenseLayer*)(model->layers[1]))->weights);

    for(int i=0; i<1e7; i++) {
        model->loadData();
        model->train();
    }
    //Matrix::inspect(((DenseLayer*)(model->layers[3]))->errors);
    //Matrix::inspect(((DenseLayer*)(model->layers[3]))->nodes);
    //Matrix::inspect(model->dataBatch[model->dataBatch.size()-1]);
    //Matrix::inspect(model->labelBatch[model->labelBatch.size()-1]);
    //Matrix::inspect(model->layers[3]->nodes);

}
