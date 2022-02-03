//
// Created by DanielSun on 11/22/2021.
//

#ifndef CUDANNGEN2_DENSEMLP_CUH
#define CUDANNGEN2_DENSEMLP_CUH
#include "Model.cuh"
#include "../utils/Matrix.cuh"
#include <vector>
#include <string>
#include <cstdio>
#include "../layers/Layer.cuh"
#include <io.h>

using namespace std;
class DenseMLP : Model{
public:
    struct Config : EngineConfig{
        int BMP_READ_SIZE = 2390;
        int INPUT_SIZE_X = 28;
        int OUTPUT_SIZE = 10;
    };
    struct Config cfg{};
    Matrix::Matrix2d* costBuffer;
    float pastCost;

    vector<Layer*> layers;
    Layer* correctOut = new Layer(cfg.OUTPUT_SIZE);
    vector<Matrix::Matrix2d*> dataBatch;
    vector<Matrix::Matrix2d*> labelBatch;
    vector<Matrix::Matrix2d*> dataset;
    vector<Matrix::Matrix2d*> labelSet;

    int calcCorrection(int success);

    void registerModel() override;
    void loadModel() override;
    void loadDataSet() override;
    void loadData() override;
    void run() override;
    void train() override;
};


#endif //CUDANNGEN2_DENSEMLP_CUH
