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
#include "models/TestModel.cuh"

#pragma comment(lib, "cublas.lib")

using namespace std;


int main(int argc, char **argv) {


    auto *model = new DenseMLP();
    model->registerModel();
    model->loadModel();
    model->loadDataSet();
    for(int i=0; i<1000; i++) {
        model->loadData();
        model->train();
    }
    Matrix::inspect(model->layers[3]->nodes);

}
