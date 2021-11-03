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
//E = (al - y) *hadmard* (sigmoidDerivative(z))
void DenseLayer::reflectOutput(Matrix::Matrix2d *correctOut) {
    copyD2D(nodes, errors);
    errors = *(*errors-correctOut)*sigmoidD(z,z);
}

