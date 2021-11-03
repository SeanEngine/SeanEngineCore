#include <cstdio>
#include "utils/Matrix.cuh"
#include <windows.h>
#include <string>
#include <iostream>
#include "cublas_v2.h"
#pragma comment(lib, "cublas.lib")

using namespace std;

int main(int argc, char** argv) {
     Matrix::Matrix2d* A, *B, *C;
     int TEST_SIZE=50;
     cudaMallocHost((void**)&A, sizeof(Matrix::Matrix2d));
     cudaMallocHost((void**)&B, sizeof(Matrix::Matrix2d));
     cudaMallocHost((void**)&C, sizeof(Matrix::Matrix2d));

     Matrix::callAllocElementD(A,TEST_SIZE,TEST_SIZE);
     Matrix::callAllocElementD(B, TEST_SIZE, TEST_SIZE);
     Matrix::callAllocElementD(C, TEST_SIZE, TEST_SIZE);

     Matrix::callAllocZero(A);
     float* buf;
     cudaMallocHost((void**)(&buf), sizeof(float));
     buf[0] = 0;
     sum(A, buf);
     //Matrix::inspect(A);
     cout<<*buf<<endl;
}
