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


int main(int argc, char **argv) {

    auto *model = new DenseMLP();
    model->registerModel();
    model->loadModel();
    model->loadDataSet();
    model->loadData();
    Matrix::inspect(model->dataBatch[0]);
    Matrix::inspect(model->labelBatch[0]);

}
