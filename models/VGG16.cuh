//
// Created by DanielSun on 1/29/2022.
//

#ifndef CUDANNGEN2_VGG16_CUH
#define CUDANNGEN2_VGG16_CUH


#include "Model.cuh"
#include <vector>
#include "../layers/Layer.cuh"

class VGG16 : public Model {
public:
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


#endif //CUDANNGEN2_VGG16_CUH
