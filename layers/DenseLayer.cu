//
// Created by DanielSun on 11/1/2021.
//

#include "DenseLayer.cuh"
#include <cassert>

string DenseLayer::getType() {
    return "DENSE";
}

//z = w * a + b , a1 = sigmoid(z)
void DenseLayer::calcActivate(Matrix::Matrix2d *prevNodes) {
    nodes = lRelu(*cross(weights, prevNodes, z) + biases, nodes);
}
//El = (al - y) *Hadamard* (sigmoidDerivative(z))
void DenseLayer::propagatingOutput(Matrix::Matrix2d *correctOut) {
    copyD2D(nodes, errors);
    errors = *(*errors-correctOut) * lReluD(z, z);
}
//El = (W(l+1))^T * E(l+1) *Hadamard* sigmoidD(z(l))
void DenseLayer::propagate(Matrix::Matrix2d *nextWeights, Matrix::Matrix2d *nextErrors) {
    errors = *cross((transpose(nextWeights,weightBuffer)),nextErrors, errors)* lReluD(z,z);
}

//add the deltas onto the recorded derivatives for averaging in batch training
//WD = El * (a(l-1))^T
void DenseLayer::recWeights(Matrix::Matrix2d *prevActivations) {
    weightDerivatives = crossA(errors, transpose(prevActivations, prevActivationBuffer), weightDerivatives);
}

//WB = El
void DenseLayer::recBias() {
    *biasDerivatives += errors;
}

void DenseLayer::applyWeights(int BATCH_SIZE, float LEARNING_RATE) {
    weights = *(*weights * WEIGHT_DECAY) - (*(*weightDerivatives * (1.0F/(float)BATCH_SIZE)) * LEARNING_RATE);
}

void DenseLayer::applyBias(int BATCH_SIZE, float LEARNING_RATE) {
    biases = *biases - (*(*biasDerivatives * (1.0F/(float)BATCH_SIZE))* LEARNING_RATE);
}

//these methods are inherited, thus pre-conditions are needed
void DenseLayer::activate(Layer *prev) {
    assert(prev->nodes!=nullptr);
    assert(prev->nodes->rowcount == this->PREV_NODE_NUMBER);
    this->calcActivate(prev->nodes);
}

void DenseLayer::propagate(Layer *prev, Layer *next) {
    assert(prev->nodes != nullptr && next->nodes != nullptr);
    assert(prev->NODE_NUMBER == this->PREV_NODE_NUMBER);
    assert(next->NODE_NUMBER == this->NEXT_NODE_NUMBER);
    //as hidden layer
    if(next->getType()=="DENSE"){
        auto* proc = (DenseLayer*)next;
        propagate(proc->weights, proc->errors);
    }

    //as output layer
    if(next->getType()=="CONTAINER"){
        propagatingOutput(next->nodes);
    }

    recWeights(prev->nodes);
    recBias();
}

void DenseLayer::learn(int BATCH_SIZE, float LEARNING_RATE) {

    applyWeights(BATCH_SIZE, LEARNING_RATE);
    applyBias(BATCH_SIZE, LEARNING_RATE);
    //reset
    Matrix::callAllocZero(weightDerivatives);
    Matrix::callAllocZero(biasDerivatives);
}

float DenseLayer::calcQuadraticCost(Matrix::Matrix2d *correctOut) {
    float ret=0;
    for(int i=0; i<nodes->rowcount; i++){
        ret += 0.5f * (float)pow(nodes->getH2D(i,0) - correctOut->getH2D(i,0),2);
    }
    return ret;
}

