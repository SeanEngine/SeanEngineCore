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
#include "models/VGG16.cuh"
#include "layers/ImageContainer.cuh"
#include "layers/ConvLayer.cuh"
#include "layers/SoftmaxLayer.cuh"
#include "layers/MaxPoolingLayer.cuh"
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

#pragma comment(lib, "cublas.lib")

using namespace std;
using namespace nvcuda;

void run(){

    auto* model = new VGG16();
    model->registerModel();
    model->loadModel();
    model->loadDataSet();
    for(int i=0; i<100; i++){
        logInfo("training batch ID: " + to_string(i),0x01);
        model->loadData();
        model->train();
        Matrix::inspect(((ConvLayer*)model->layers[14])->output);
    }
}

void GEMMBench(){
    Matrix::Matrix2d* A, *B, *C;
    A = Matrix::callAllocElementD(64,27);
    B = Matrix::callAllocElementD(27,50176);
    C = Matrix::callAllocElementD(64,50176);

    Matrix::callAllocRandom(A);
    Matrix::callAllocRandom(B);
    Matrix::callAllocConst(C,1);

    LARGE_INTEGER beg;
    LARGE_INTEGER end;
    LARGE_INTEGER frq;

    QueryPerformanceFrequency(&frq);
    QueryPerformanceCounter(&beg);
    crossA(A,B,C);
    cout<<endl;
    QueryPerformanceCounter(&end);
    cout<<end.QuadPart - beg.QuadPart<<endl;

    QueryPerformanceCounter(&beg);
    Matrix::callCrossPrefetching(A,B,C);
    cout<<endl;
    QueryPerformanceCounter(&end);
    cout<<end.QuadPart - beg.QuadPart<<endl;
}

int main(int argc, char **argv) {
    run();
}
