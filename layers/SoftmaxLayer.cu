//
// Created by DanielSun on 12/8/2021.
//

#include "SoftmaxLayer.cuh"

string SoftmaxLayer::getType() {
    return "SOFTMAX";
}

void SoftmaxLayer::calcActivate(Matrix::Matrix2d *prevNodes) {
    z = *cross(weights, prevNodes, z) + biases;
    nodes = softmax(z,nodes, nullptr);
}

void SoftmaxLayer::propagatingOutput(Matrix::Matrix2d *correctOut) {
    errors = softmaxD(nodes, correctOut, errors);
}

