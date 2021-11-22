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
#include <io.h>

using namespace std;
class DenseMLP : Model{
public:
    struct Config : Engine::EngineConfig{
        int BMP_READ_SIZE = 2390;
        int BMP_READ_DIM = 28;
        int OUTPUT_SIZE = 10;
    };
    struct Config cfg{};

    vector<Matrix::Matrix2d*> dataset;
    vector<Matrix::Matrix2d*> labelSet;

    void loadDataSet() override;
    void execute() override;
    void loadData() override;
    void loadModel() override;
    void train() override;
    void unloadData() override;
};


#endif //CUDANNGEN2_DENSEMLP_CUH
