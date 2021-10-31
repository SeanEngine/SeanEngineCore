#include <cstdio>
#include "utils/Matrix.cuh"
#include <windows.h>
#include <string>
#include <iostream>
#include "cublas_v2.h"
#pragma comment(lib, "cublas.lib")

int main(int argc, char** argv) {
    Matrix::Matrix2d *A, *B,*C;
    cudaMallocHost(reinterpret_cast<void **>(&A), sizeof(Matrix::Matrix2d));
    cudaMallocHost(reinterpret_cast<void **>(&B), sizeof(Matrix::Matrix2d));
    cudaMallocHost(reinterpret_cast<void **>(&C), sizeof(Matrix::Matrix2d));

    Matrix::callAllocElementD(A, 1000, 1000);
    Matrix::callAllocElementD(B, 1000, 1000);
    Matrix::callAllocElementD(C, 1000, 1000);

    Matrix::callAllocRandom(A);
    _sleep(1000);
    Matrix::callAllocRandom(B);
    LARGE_INTEGER cpuFre;
    LARGE_INTEGER begin;
    LARGE_INTEGER end;

    QueryPerformanceFrequency(&cpuFre);
    QueryPerformanceCounter(&begin);
    Matrix::callCrossCompOpt(A, B, C);
    QueryPerformanceCounter(&end);
    std::cout <<"CompOpt : "<< std::to_string(((double) end.QuadPart - (double) begin.QuadPart)/1e7)<<std::endl;
    //Matrix::inspect(C);
    std::cout << "--------------------------------------"<<std::endl;
    QueryPerformanceCounter(&begin);
    cross(A, B, C);
    QueryPerformanceCounter(&end);
    std::cout <<"Tiling : "<< std::to_string(((double) end.QuadPart - (double) begin.QuadPart)/1e7)<<std::endl;
    //Matrix::inspect(C);
    std::cout << "--------------------------------------"<<std::endl;
    QueryPerformanceCounter(&begin);
    Matrix::callCrossPOlD(A,B,C);
    QueryPerformanceCounter(&end);
    std::cout << "Standard : "<<std::to_string(((double) end.QuadPart - (double) begin.QuadPart)/1e7) <<std::endl;
    //Matrix::inspect(C);
    /*
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, A->rowcount, B->colcount, A->colcount,
                &al, A->elements, A->rowcount, B->elements, B->colcount, &bet, C->elements, A->rowcount);
                */

    cudaFree(A->elements);
    cudaFree(B->elements);
    cudaFree(C->elements);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}
