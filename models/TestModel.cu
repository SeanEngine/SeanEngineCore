//
// Created by DanielSun on 11/25/2021.
//

#include "TestModel.cuh"
#include "../layers/DenseLayer.cuh"

void TestModel::registerModel() {
    cudaMallocHost((void **) &costBuffer, sizeof(Matrix::Matrix2d));
    Matrix::callAllocElementD(costBuffer, cfg.OUTPUT_SIZE, 1);
    logInfo("===========< REGISTERING : TestModel >============", 0x05);
    layers.push_back(new Layer(cfg.INPUT_SIZE));
    layers.push_back(new DenseLayer(3, 3, 2, 1));
    layers.push_back(new DenseLayer(2, 3, 2, 2));
}

void TestModel::loadModel() {
    for (Layer *layer: layers) {
        if (layer->getType() == "DENSE") {
            auto *temp = (DenseLayer *) layer;
            Matrix::callAllocRandom(temp->weights);
            Matrix::callAllocRandom(temp->biases);
            logInfo("layer: " + layer->getType() + " random allocated");
        }
    }
}

void TestModel::loadDataSet() {
    Matrix::Matrix2d *in1;
    Matrix::Matrix2d *in2;
    Matrix::Matrix2d *lab1;
    Matrix::Matrix2d *lab2;
    cudaMallocHost((void **) &in1, sizeof(Matrix::Matrix2d));
    cudaMallocHost((void **) &in2, sizeof(Matrix::Matrix2d));
    cudaMallocHost((void **) &lab1, sizeof(Matrix::Matrix2d));
    cudaMallocHost((void **) &lab2, sizeof(Matrix::Matrix2d));
    Matrix::callAllocElementD(in1, 3, 1);
    Matrix::callAllocElementD(in2, 3, 1);
    Matrix::callAllocElementD(lab1, 2, 1);
    Matrix::callAllocElementD(lab2, 2, 1);
    dataBatch.push_back(in1);
    dataBatch.push_back(in2);
    labelBatch.push_back(lab1);
    labelBatch.push_back(lab2);
}

void TestModel::loadData() {
    float input1[] = {1, 0, 0};
    float input2[] = {0, 0, 1};
    float output1[] = {0, 1};
    float output2[] = {1, 0};
    cudaMemcpy(dataBatch[0]->elements, input1, sizeof(float)*3, cudaMemcpyHostToDevice);
    cudaMemcpy(dataBatch[1]->elements, input2, sizeof(float)*3, cudaMemcpyHostToDevice);
    cudaMemcpy(labelBatch[0]->elements, output1, sizeof(float)*2, cudaMemcpyHostToDevice);
    cudaMemcpy(labelBatch[1]->elements, output2, sizeof(float)*2, cudaMemcpyHostToDevice);
}

void TestModel::train() {
    pastCost = 0;
    for (int trial = 0; trial < cfg.TRAIN_BATCH_SIZE; trial++) {
        layers[0]->nodes = flattern(dataBatch[trial]);
        for (int i = 1; i < layers.size(); i++) {
            layers[i]->activate(layers[i - 1]);
        }

        costBuffer = *(*(*copyD2D(layers[layers.size()-1]->nodes, costBuffer) - labelBatch[trial])^2)*0.5;
        correctOut->nodes = labelBatch[trial];
        float cost = sumC(costBuffer);
        pastCost += cost;

        for (int i = (int)layers.size()-1; i > 0; i--) {
            layers[i]->propagate(layers[i - 1], i+1 < layers.size()? layers[i+1] : correctOut);
        }
    }
    Matrix::inspect(((DenseLayer*)layers[2])->weightDerivatives);
    Matrix::inspect(((DenseLayer*)layers[2])->biasDerivatives);
    for (int i = (int)layers.size()-1; i > 0; i--) {
        layers[i]->learn(cfg.TRAIN_BATCH_SIZE, cfg.LEARNING_RATE);
    }
    logInfo("Batch trained with cost: " + to_string(pastCost/(float)cfg.TRAIN_BATCH_SIZE));
}

void TestModel::run() {

}
