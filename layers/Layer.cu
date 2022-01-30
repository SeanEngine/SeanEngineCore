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

Layer* Layer::bindPrev(Layer* prev) {
    this->prevLayer = prev;
    return this;
}

Layer* Layer::bindNext(Layer *next) {
    this->nextLayer = next;
    return this;
}

Layer* Layer::bind(Layer* prev, Layer *next) {
    this->prevLayer = prev;
    this->nextLayer = next;
    return this;
}

void Layer::activate() {
    activate(this->prevLayer);
}

void Layer::propagate() {
    propagate(this->prevLayer, this->nextLayer);
}

void Layer::randomInit() {

}
