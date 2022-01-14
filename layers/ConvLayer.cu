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

void ConvLayer::activate(Matrix::Tensor3d *prevFeatures) {
    assert(prevFeatures->rowcount == prevFeatures->colcount);
    assert(prevFeatures->rowcount == paddedFeature->rowcount - (filters->rowcount/2)*2);
    pad3D(prevFeatures, paddedFeature, (filters->rowcount/2)*2);
    conv2D(paddedFeature, filters, z, stride, filterBuffer, featureMapBuffer, zBuffer, filterBiases);
    lRelu(z, output);
}

void ConvLayer::propagate(Matrix::Matrix2d *prevErrors, Matrix::Matrix2d *prevZBuffer) {
    assert(prevErrors->rowcount == paddedFeature->depthCount && prevErrors->colcount == (
            pow(paddedFeature->rowcount-(filters->rowcount/2)*2,2)
        ));
    unsigned int padSize = (filters->rowcount/2)*2;
    propaBuffer = cross(transpose(filterBuffer, filterBufferTrans), errors, propaBuffer);
    col2img2D(prevErrors, propaBuffer, output->rowcount, paddedFeature->rowcount-2*padSize,filters->rowcount, stride, padSize);
    *prevErrors * lReluD(prevZBuffer, prevZBuffer);
}