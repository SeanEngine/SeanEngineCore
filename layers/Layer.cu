//
// Created by DanielSun on 11/1/2021.
//

#include "Layer.cuh"

string Layer::getType() {
    return "CONTAINER";
}

void Layer::activate(Layer *prevLayer) {
    copyD2D(prevLayer->nodes, this->nodes);
}

void Layer::propagate(Layer *prev, Layer *next) {

}

void Layer::learn(int BATCH_SIZE, float LEARNING_RATE) {

}
