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

    auto *model = new DenseMLP();
    model->registerModel();
    model->loadModel();
    model->loadDataSet();
    model->loadData();

    auto* label = new Layer(10);
    label->nodes = model->labelBatch[0];
    cout<<gradCheck(model->layers, model->dataBatch[0], label);
/*
    for(int i=0; i<1000; i++) {
        model->loadData();
        model->train();
    }
    */

    //Matrix::inspect(model->layers[3]->nodes);


}
