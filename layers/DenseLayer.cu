//
// Created by DanielSun on 11/1/2021.
//

#include "DenseLayer.cuh"

string DenseLayer::getType() {
    return "DENSE";
}

//z = w * a + b , a1 = sigmoid(z)
void DenseLayer::activate(Matrix::Matrix2d *prevNodes) {
    nodes = sigmoid(*cross(weights, prevNodes, z) + biases, nodes);
}
//El = (al - y) *Hadamard* (sigmoidDerivative(z))
void DenseLayer::reflectOutput(Matrix::Matrix2d *correctOut) {
    copyD2D(nodes, errors);
    errors = *(*errors-correctOut)*sigmoidD(z,z);
}
//El = (W(l+1))^T * E(l+1) *Hadamard* sigmoidD(z(l))
void DenseLayer::reflect(Matrix::Matrix2d *nextWeights, Matrix::Matrix2d *nextBias, Matrix::Matrix2d *nextErrors) {
    errors = *cross((transpose(nextWeights,weightBuffer)),nextErrors, errors)* sigmoidD(z,z);
}

//add the deltas onto the recorded derivatives for averaging in batch training
//WD = El * (a(l-1))^T
void DenseLayer::recWeights(Matrix::Matrix2d *prevActivations) {
    weightDerivatives = crossA(errors, transpose(prevActivations, prevActivationBuffer), weightDerivatives);
}

//WB = El
void DenseLayer::recBias() {
    biasDerivatives = *biasDerivatives + errors;
}

void DenseLayer::applyWeights(int BATCH_SIZE, float LEARNING_RATE) {
    weights = *weights - *(*weightDerivatives * (1.0F/BATCH_SIZE)) * LEARNING_RATE;
}
