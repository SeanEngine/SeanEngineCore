//
// Created by DanielSun on 11/25/2021.
//

#ifndef CUDANNGEN2_TESTMODEL_CUH
#define CUDANNGEN2_TESTMODEL_CUH


#include "Model.cuh"
#include "../utils/Matrix.cuh"
#include "../layers/Layer.cuh"
#include "vector"

class TestModel : Model{
public:
    struct Config : Engine::EngineConfig{
        int INPUT_SIZE = 3;
        int OUTPUT_SIZE = 2;
    };
    struct Config cfg{};
    Matrix::Matrix2d* costBuffer;
    float pastCost;

    vector<Layer*> layers;
    Layer* correctOut = new Layer(cfg.OUTPUT_SIZE);
    vector<Matrix::Matrix2d*> dataBatch;
    vector<Matrix::Matrix2d*> labelBatch;

    void registerModel() override;

    void loadModel() override;

    void loadDataSet() override;

    void loadData() override;

    void train() override;

    void run() override;
};


#endif //CUDANNGEN2_TESTMODEL_CUH
