//
// Created by DanielSun on 1/14/2022.
//

#include "MaxPoolingLayer.cuh"
#include "../utils/NeuralUtils.cuh"
#include "ConvLayer.cuh"

string MaxPoolingLayer::getType() {
    return "MAXPOOL";
}

void MaxPoolingLayer::activate(Layer *prevLayer) {
    if(prevLayer->getType() == "CONV2D"){
        auto* proc = (ConvLayer*)prevLayer;
        Matrix::callAllocZero(record);
        NeuralUtils::callMaxPooling(proc->output, record, output, stride);
        NeuralUtils::callMaxPooling(proc->z, z, stride);
    }
}

void MaxPoolingLayer::propagate(Layer *prev, Layer *next) {

    if(prev->getType() == "CONV2D"){
        auto* proc = (ConvLayer*)prev;
        NeuralUtils::callInvertMaxPool(errors, record, proc->errorsJunction3D, stride);
    }
}

void MaxPoolingLayer::learn(int BATCH_SIZE, float LEARNING_RATE) {
    //pass, this layer does not learn anything
}
