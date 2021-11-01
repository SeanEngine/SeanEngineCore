//
// Created by DanielSun on 11/1/2021.
//

#include "DenseLayer.cuh"

string DenseLayer::getType() {
    return "DENSE";
}

//z = w * a + b , a1 = sigmoid(z)
void DenseLayer::activate(Matrix::Matrix2d *prevNodes) {
    this->z = *cross(this->weights, prevNodes, this->z) + this->biases;
    this->nodes = sigmoid(this->z, this->nodes);
}
//E = (al - y) *hadmard* (sigmoidDerivative(z))
void DenseLayer::reflectOutput(Matrix::Matrix2d *correctOut) {
    copyD2D(this->nodes, this->errors);
    this->errors = *(*this->errors-correctOut)*sigmoidD(this->z,this->z);
}

