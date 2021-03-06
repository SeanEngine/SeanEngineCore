//
// Created by DanielSun on 11/22/2021.
//

#include "DenseMLP.cuh"
#include "../io/Reader.cuh"
#include "../utils/logger.cuh"
#include "../utils/Matrix.cuh"
#include "../layers/DenseLayer.cuh"
#include "../layers/SoftmaxLayer.cuh"
#include <random>

int readDataset(const string& path0, vector<Matrix::Matrix2d*>* data, vector<Matrix::Matrix2d*>* label,
                 DenseMLP::Config cfg, int labelIndex) {

    vector<string> temp = Reader::getDirFiles(path0);
    for(int i=0; i < temp.size(); i++){
        auto* labelElement = Matrix::callAllocElementD(cfg.OUTPUT_SIZE,1);
        Matrix::callAllocZero(labelElement);

        auto* dataElement =  Matrix::callAllocElementD(cfg.INPUT_SIZE_X, cfg.INPUT_SIZE_X);

        labelElement->setH2D(labelIndex, 1.0f);
        label->push_back(labelElement);
        data->push_back(dataElement);
    }
    Reader::readImgGray(cfg.CPU_THREADS, dim3i(cfg.INPUT_SIZE_X,cfg.INPUT_SIZE_X), &temp, data);
    logInfo("DATASET > read " + to_string(temp.size())+ " files for label : " + to_string(labelIndex));
    return (int)temp.size();
}

void DenseMLP::registerModel() {
     costBuffer = Matrix::callAllocElementD(cfg.OUTPUT_SIZE, 1);
     logInfo("===========< REGISTERING : DenseMLP >============",0x05);
     layers.push_back(new Layer(784));  //input layer
     layers.push_back(new DenseLayer(128, 784, 16, 1));
     layers.push_back(new DenseLayer(16, 128, 10, 2));
     layers.push_back(new SoftmaxLayer(10, 16, 10, 3));
}


void DenseMLP::loadModel() {
    logInfo("===========< LOADING : DenseMLP >============",0x05);
     if(cfg.LOAD_MODEL_FROM_SAV){
         //....
         return;
     }
    for (Layer* layer : layers){
        layer->randomInit();
    }
}

void DenseMLP::loadDataSet() {
     string path0 = DenseMLP::cfg.TRAIN_DATA_PATH;
     int count = 0;
     for(int i=0; i< 10; i++){
         count += readDataset(path0 + "\\" + to_string(i), &dataset, &labelSet, cfg, i);
     }

    for (int i=0; i< cfg.TRAIN_BATCH_SIZE; i++){
        auto* data = Matrix::callAllocElementD( cfg.INPUT_SIZE_X, cfg.INPUT_SIZE_X);
        dataBatch.push_back(data);
        auto* label = Matrix::callAllocElementD(cfg.OUTPUT_SIZE,1);
        labelBatch.push_back(label);
    }
}

void DenseMLP::run() {
    //forward feeding
    for (int i = 1; i < layers.size(); i++) {
        layers[i]->activate(layers[i - 1]);
    }
}

void DenseMLP::loadData() {
    random_device rd;
    default_random_engine gen = default_random_engine(rd());
    uniform_int_distribution<int> dis(0,(int)dataset.size()-1);
    for(int i=0; i<cfg.TRAIN_BATCH_SIZE; i++){
        int index = dis(gen);
        copyH2D(dataset[index], dataBatch[i]);
        copyH2D(labelSet[index], labelBatch[i]);
    }
}

int DenseMLP::calcCorrection(int success){
    //calculate correction
    int maxIndex1 = 0, maxIndex2 = 0;
    auto* debug = Matrix::callAllocElementH(DenseMLP::cfg.OUTPUT_SIZE, 1);
    cudaMemcpy(debug->elements, layers[3]->nodes->elements, sizeof(float) * cfg.OUTPUT_SIZE, cudaMemcpyDeviceToHost);
    for(int i=0; i< cfg.OUTPUT_SIZE; i++) {
        maxIndex1 = *(debug->elements + i) > *(debug->elements + maxIndex1) ? i : maxIndex1;
    }
    cudaMemcpy(debug->elements, correctOut->nodes->elements, sizeof(float) * cfg.OUTPUT_SIZE, cudaMemcpyDeviceToHost);
    for(int i=0; i< cfg.OUTPUT_SIZE; i++){
        maxIndex2 = *(debug->elements + i) > *(debug->elements + maxIndex2) ? i: maxIndex2;
    }
    cudaFreeHost(debug->elements);
    cudaFreeHost(debug);
    return maxIndex1 == maxIndex2 ? success+1 : success;
}

void DenseMLP::train() {
    int success=0;
    pastCost=0;
    for (int trial = 0; trial < cfg.TRAIN_BATCH_SIZE; trial++) {
        layers[0]->nodes = flatten(dataBatch[trial]);

        //forward feeding
        for (int i = 1; i < layers.size(); i++) {
            layers[i]->activate(layers[i - 1]);
        }

        correctOut->nodes = labelBatch[trial];
        //calculate cost
        costBuffer = softMaxL(layers[layers.size()-1]->nodes, labelBatch[trial], costBuffer);
        float cost = sumH(costBuffer);
        pastCost+=cost;

        //calculate correction
        success = calcCorrection(success);

        //back propagate
        for (int i = (int)layers.size()-1; i > 0 ; i--) {
            layers[i]->propagate(layers[i - 1],i+1 < layers.size()? layers[i+1] : correctOut);
        }
    }


    //apply changes (errors)
    for (int i = (int)layers.size()-1; i > 0; i--) {
        layers[i]->learn(cfg.TRAIN_BATCH_SIZE, cfg.LEARNING_RATE);
    }
    logInfo("Batch trained with cost: " + to_string(pastCost/(float)cfg.TRAIN_BATCH_SIZE) +
    " success rate: " + to_string(success));
}