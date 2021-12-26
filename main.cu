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
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

#pragma comment(lib, "cublas.lib")

using namespace std;
using namespace nvcuda;

void run(){

    auto *model = new DenseMLP();
    model->registerModel();
    model->loadModel();
    model->loadDataSet();
    model->loadData();
    //Matrix::inspect(((DenseLayer*)(model->layers[1]))->weights);

    for(int i=0; i<1e4; i++) {
        model->loadData();
        model->train();
    }

    int success = 0;
    for(int trial=0; trial<60000; trial++){
        Matrix::Matrix2d* data = model->dataset[trial];
        Matrix::Matrix2d* label = model->labelSet[trial];
        model->layers[0]->nodes = flattern(data);
        model->run();

        int maxIndex1 = 0, maxIndex2 = 0;
        Matrix::Matrix2d* debug;
        cudaMallocHost((void**)&debug, sizeof(Matrix::Matrix2d));
        Matrix::callAllocElementH(debug, 10, 1);
        cudaMemcpy(debug->elements, model->layers[3]->nodes->elements, sizeof(float) * 10, cudaMemcpyDeviceToHost);
        for(int i=0; i< 10; i++) {
            maxIndex1 = debug->elements[i] > debug->elements[maxIndex1] ? i : maxIndex1;
        }
        cudaMemcpy(debug->elements, label->elements, sizeof(float) * 10, cudaMemcpyDeviceToHost);
        for(int i=0; i< 10; i++){
            maxIndex2 =  debug->elements[i] > debug->elements[maxIndex2] ? i : maxIndex2;
        }
        success = maxIndex1 == maxIndex2 ? success+1 : success;
    }

    cout<<success<<endl;
}

int main(int argc, char **argv) {

    run();

    /*
    auto *model = new DenseMLP();
    model->registerModel();
    model->loadModel();
    model->loadDataSet();

    Matrix::Matrix2d* mat0 = model->dataset[0];
    Matrix::Matrix2d* mat1 = model->dataset[1];
    Matrix::Matrix2d* mat2;
    cudaMallocHost(&mat2, sizeof(Matrix::Matrix2d));
    Matrix::callAllocElementD(mat2,28,28);
    Matrix::inspect(mat0);
    cout<<endl;
    Matrix::inspect(mat1);
    cout<<endl;
    Matrix::inspect(cross(mat0,mat1,mat2));
    */
}
