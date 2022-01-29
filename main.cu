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

    for(int i=0; i<1e4*3; i++) {
        model->loadData();
        model->train();
    }
}

int main(int argc, char **argv) {
    auto* model = new VGG16();
    model->registerModel();
    run();
}
