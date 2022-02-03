//
// Created by DanielSun on 1/14/2022.
//

#ifndef CUDANNGEN2_MAXPOOLINGLAYER_CUH
#define CUDANNGEN2_MAXPOOLINGLAYER_CUH

#include "../utils/Matrix.cuh"
#include "Layer.cuh"
#include "../utils/logger.cuh"

class MaxPoolingLayer : public Layer{

public:
    Matrix::Tensor3d *output, *errors, *record, *z;
    Matrix::Matrix2d *zJunction1D{}, *zJunction2D{}, *errorJunction1D{}, *errorJunction2D{};
    int stride;

    MaxPoolingLayer(dim3 inputSize, int stride) : Layer((inputSize.z) * (inputSize.y/stride)*(inputSize.x/stride)) {
        output = Matrix::callAllocElementD(inputSize.z, inputSize.y/stride, inputSize.x/stride);
        errors =  Matrix::callAllocElementD(inputSize.z, inputSize.y/stride, inputSize.x/stride);
        record = Matrix::callAllocElementD(inputSize.z, inputSize.y, inputSize.x);
        z = Matrix::callAllocElementD(inputSize.z, inputSize.y/stride, inputSize.x/stride);

        cudaMallocHost(&zJunction1D, sizeof(float)*z->elementCount);
        cudaMallocHost(&zJunction2D, sizeof(float)*z->elementCount);
        cudaMallocHost(&errorJunction1D, sizeof(float)*errors->elementCount);
        cudaMallocHost(&errorJunction2D, sizeof(float)*errors->elementCount);

        //these are just junctions to other types of layers
        zJunction1D->index(z->elementCount, 1, z->elements);
        zJunction2D->index(z->depthCount, z->colcount * z->rowcount, z->elements);
        errorJunction1D->index(errors->elementCount, 1, errors->elements);
        errorJunction2D->index(errors->depthCount, errors->colcount * errors->rowcount, errors->elements);
        this->stride = stride;

        cudaMallocHost(&this->nodes, sizeof(Matrix::Matrix2d));
        this->nodes->index(output->elementCount, 1, output->elements);

        logInfo("MAX POOLING : " + record->toString() + " -> " + output->toString());
    }

    string getType() override;

    void activate(Layer *prevLayer) override;
    void propagate(Layer *prev, Layer *next) override;
    void learn(int BATCH_SIZE, float LEARNING_RATE) override;
};


#endif //CUDANNGEN2_MAXPOOLINGLAYER_CUH
