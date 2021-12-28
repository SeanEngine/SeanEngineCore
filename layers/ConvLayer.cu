//
// Created by DanielSun on 12/28/2021.
//

#include "ConvLayer.cuh"

void ConvLayer::activate(Layer *prevLayer) {
    Layer::activate(prevLayer);
}

void ConvLayer::propagate(Layer *prev, Layer *next) {
    Layer::propagate(prev, next);
}

void ConvLayer::learn(int BATCH_SIZE, float LEARNING_RATE) {
    Layer::learn(BATCH_SIZE, LEARNING_RATE);
}

string ConvLayer::getType() {
    return "CONV2D";
}
