//
// Created by DanielSun on 12/8/2021.
//

#ifndef CUDANNGEN2_SOFTMAXLAYER_CUH
#define CUDANNGEN2_SOFTMAXLAYER_CUH
#include "Layer.cuh"
#include "DenseLayer.cuh"

class SoftmaxLayer : DenseLayer{
public:
    SoftmaxLayer(int nodeNumber, int prevNodeNumber, int nextNodeNumber, int layerId)
         : DenseLayer(nodeNumber, prevNodeNumber, nextNodeNumber, layerId) {

    }

    string getType() override;

    //this is change to the calculations for softmax activation
    void calcActivate(Matrix::Matrix2d* prevNodes) override;

    void propagatingOutput(Matrix::Matrix2d *correctOut) override;
};


#endif //CUDANNGEN2_SOFTMAXLAYER_CUH
