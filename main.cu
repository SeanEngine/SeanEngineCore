#include <cstdio>
#include "utils/Matrix.cuh"
#include <windows.h>
#include <string>
#include <iostream>
#include "io/Reader.cuh"
#include "cublas_v2.h"
#include "execution/ThreadControls.h"
#include "models/DenseMLP.cuh"

#pragma comment(lib, "cublas.lib")

using namespace std;


int main(int argc, char **argv) {

    auto *model = new DenseMLP();
    model->registerModel();
    model->loadModel();
    model->loadDataSet();
    Matrix::inspect(model->dataset[20000]);
}
