//
// Created by DanielSun on 12/28/2021.
//

#include "ConvLayer.cuh"
#include "MaxPoolingLayer.cuh"

void ConvLayer::activate(Layer *prevLayer) {
    if(prevLayer->getType() == "CONV2D"){
        activate(((ConvLayer*)prevLayer)->output);
    }
    if(prevLayer->getType() == "MAXPOOL"){
        activate(((MaxPoolingLayer*)prevLayer)->output);
    }
}

void ConvLayer::propagate(Layer *prev, Layer *next) {
    if(prevLayer->getType() == "CONV2D"){
        propagate(((ConvLayer*)prevLayer)->errors, ((ConvLayer*)prevLayer)->zBuffer);
    }
    if(prevLayer->getType() == "MAXPOOL"){
        propagate(((MaxPoolingLayer*)prevLayer)->errorJunction2D, ((MaxPoolingLayer*)prevLayer)->zJunction2D);
    }
}

void ConvLayer::learn(int BATCH_SIZE, float LEARNING_RATE) {
    applyFilters(BATCH_SIZE, LEARNING_RATE);
    applyBiases(BATCH_SIZE, LEARNING_RATE);
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

//this method actually calculate the errors for the previous layer
void ConvLayer::propagate(Matrix::Matrix2d *prevErrors, Matrix::Matrix2d *prevZBuffer) {
    assert(prevErrors->rowcount == paddedFeature->depthCount && prevErrors->colcount == (
            pow(paddedFeature->rowcount-(filters->rowcount/2)*2,2)
        ));
    unsigned int padSize = (filters->rowcount/2)*2;
    propaBuffer = cross(transpose(filterBuffer, filterBufferTrans), errors, propaBuffer);
    col2img2D(prevErrors, propaBuffer, output->rowcount, paddedFeature->rowcount-2*padSize,filters->rowcount, stride, padSize);
    *prevErrors * lReluD(prevZBuffer, prevZBuffer);
}

void ConvLayer::recFilters() const {
     crossA(errors, transpose(featureMapBuffer, featureMapTrans), filterDBuffer);
}

void ConvLayer::recBiases() {
     filterBiasD = rowReduce(errors, filterBiasD, filterPropagateBuffer);
}

void ConvLayer::applyFilters(int BATCH_SIZE, float LEARNING_RATE) {
     filters = (Matrix::Tensor4d*)(*filters - (*filterD * (LEARNING_RATE / (float)BATCH_SIZE)));
}

void ConvLayer::applyBiases(int BATCH_SIZE, float LEARNING_RATE) {
     filterBiases = *filterBiases - (*filterBiasD * (LEARNING_RATE / (float)BATCH_SIZE));
}

void ConvLayer::randomInit() {
    Matrix::callAllocRandom(filters);
    Matrix::callAllocRandom(filterBiases);
}
