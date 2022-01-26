//
// Created by DanielSun on 1/14/2022.
//

#ifndef CUDANNGEN2_CSPDARKNET53_CUH
#define CUDANNGEN2_CSPDARKNET53_CUH

#include "Model.cuh"
#include "../layers/Layer.cuh"
#include "../utils/logger.cuh"
#include <vector>

class CSPDarknet53 : Model{

    vector<Layer*> layers;
    vector<Matrix::Matrix2d*> dataBatch;
    vector<Matrix::Matrix2d*> labelBatch;
    vector<Matrix::Matrix2d*> dataset;
    vector<Matrix::Matrix2d*> labelSet;

    void registerModel() override;
    void loadModel() override;
    void loadDataSet() override;
    void loadData() override;
    void train() override;
    void run() override;
};


#endif //CUDANNGEN2_CSPDARKNET53_CUH
