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
#include "debug_tools/GradientCheck.cuh"

#pragma comment(lib, "cublas.lib")

using namespace std;

int main(int argc, char **argv) {

    Matrix::Matrix2d* mat1;
    float * buffer;
    cudaMallocHost((void**)&mat1, sizeof(Matrix::Matrix2d));
    cudaMalloc((void**)&buffer, sizeof(float)*1028);
    Matrix::callAllocElementD(mat1, 1, 569);

    Matrix::callAllocRandom(mat1);
    NeuralUtils::callSoftMax(mat1, mat1, buffer);
    cout<<sumH(mat1)<<endl;
/*
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
    //Matrix::inspect(((DenseLayer*)(model->layers[1]))->weights);
    //Matrix::inspect(model->dataBatch[model->dataBatch.size()-1]);
    //Matrix::inspect(model->labelBatch[model->labelBatch.size()-1]);
    //Matrix::inspect(model->layers[3]->nodes);
*/
}
