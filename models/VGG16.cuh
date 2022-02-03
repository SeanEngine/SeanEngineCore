//
// Created by DanielSun on 1/29/2022.
//

#ifndef CUDANNGEN2_VGG16_CUH
#define CUDANNGEN2_VGG16_CUH


#include "Model.cuh"
#include <vector>
#include "../layers/Layer.cuh"
#include "../io/Reader.cuh"

class VGG16 : public Model {
public:
    struct ModelConfig : public EngineConfig{
         dim3i DATA_SIZE = dim3i(224,224,3);
         int OUTPUT_SIZE = 10;
    };
    ModelConfig cfg = *new ModelConfig;
    vector<Layer*> layers;
    vector<Matrix::Tensor3d*> dataBatch;
    vector<Matrix::Matrix2d*> labelBatch;
    vector<Matrix::Tensor3d*> dataset;
    vector<Matrix::Matrix2d*> labelSet;

    int calcCorrection(int success, Layer* correct);

    void registerModel() override;
    void loadModel() override;
    void loadDataSet() override;
    void loadData() override;
    void train() override;
    void run() override;
};


#endif //CUDANNGEN2_VGG16_CUH
