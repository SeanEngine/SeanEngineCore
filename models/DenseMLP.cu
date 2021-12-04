//
// Created by DanielSun on 11/22/2021.
//

#include "DenseMLP.cuh"
#include "../io/Reader.cuh"
#include "../utils/logger.cuh"
#include "../utils/Matrix.cuh"
#include "../layers/DenseLayer.cuh"
#include <random>

int readDataset(const string& path0, vector<Matrix::Matrix2d*>* data, vector<Matrix::Matrix2d*>* label,
                 DenseMLP::Config cfg, int labelIndex, int count) {
    string* paths;
    vector<Matrix::Matrix2d *> buf;
    unsigned char* buffer;
    unsigned char* bufCuda;
    vector<string> temp = Reader::getDirFiles(path0);

    cudaMallocHost((void**)&paths, sizeof(string)*temp.size());
    cudaMallocHost((void**)&buffer, sizeof(char)*cfg.CPU_THREADS*cfg.BMP_READ_SIZE);
    cudaMalloc((void**)&bufCuda, sizeof(char)*cfg.CPU_THREADS*cfg.BMP_READ_SIZE);
    for(int i=1; i< temp.size(); ++i){
        paths[i-1] = temp[i];
    }

    for (int i=0;i< temp.size();i++) {

        Matrix::Matrix2d* matData;
        cudaMallocHost(&matData,sizeof(Matrix::Matrix2d));
        Matrix::callAllocElementH(matData, cfg.INPUT_SIZE_X, cfg.INPUT_SIZE_X);
        Matrix::Matrix2d* matLabel;

        cudaMallocHost(&matLabel,sizeof(Matrix::Matrix2d));
        Matrix::callAllocElementH(matLabel, cfg.OUTPUT_SIZE,1);
        Matrix::callAllocZero(matLabel);
        matLabel->elements[labelIndex] = 1.0f;

        (*label).push_back(matLabel);
        (*data).push_back(matData);
    }

    for (int i=0;i< cfg.CPU_THREADS;i++) {
        Matrix::Matrix2d* matT;
        cudaMallocHost(&matT,sizeof(Matrix::Matrix2d));
        Matrix::callAllocElementD(matT, cfg.INPUT_SIZE_X, cfg.INPUT_SIZE_X);
        buf.push_back(matT);
    }
    int index = 0;
    while(index < temp.size()){
        int threads = temp.size()-index < cfg.CPU_THREADS ? (int)temp.size()-index : cfg.CPU_THREADS;
        Reader::readBMPFiles(threads, paths, cfg.BMP_READ_SIZE, buffer, bufCuda, data, &buf, Reader::READ_GRAY, index, count);
        index+=cfg.CPU_THREADS;
    }

    cudaFreeHost(buffer);
    cudaFreeHost(paths);
    cudaFree(bufCuda);

    logInfo("DATASET > read " + to_string(temp.size())+ " files for label : " + to_string(labelIndex));
    return (int)temp.size();
}

void DenseMLP::registerModel() {
     cudaMallocHost((void**)&costBuffer, sizeof(Matrix::Matrix2d));
     Matrix::callAllocElementD(costBuffer, cfg.OUTPUT_SIZE, 1);
     logInfo("===========< REGISTERING : DenseMLP >============",0x05);
     layers.push_back(new Layer(784));  //input layer
     layers.push_back(new DenseLayer(16, 784, 16, 1));
     layers.push_back(new DenseLayer(16, 16, 10, 2));
     layers.push_back(new DenseLayer(10, 16, 10, 3));
}


void DenseMLP::loadModel() {
    logInfo("===========< LOADING : DenseMLP >============",0x05);
     if(cfg.LOAD_MODEL_FROM_SAV){
         //....
         return;
     }
    for (Layer* layer : layers){
        if(layer->getType() == "DENSE"){
            auto* temp = (DenseLayer*)layer;
            Matrix::callAllocRandom(temp->weights);
            Matrix::callAllocRandom(temp->biases);
            logInfo("layer: " + layer->getType() + " random allocated");
        }
    }
}

void DenseMLP::loadDataSet() {
     string path0 = DenseMLP::cfg.TRAIN_DATA_PATH;
     int count = 0;
     for(int i=0; i< 10; i++){
         count += readDataset(path0 + "\\" + to_string(i), &dataset, &labelSet, cfg, i, count);
     }

    for (int i=0; i< cfg.TRAIN_BATCH_SIZE; i++){
        Matrix::Matrix2d* data, *label;
        cudaMallocHost((void**)&data, sizeof(Matrix::Matrix2d));
        cudaMallocHost((void**)&label, sizeof(Matrix::Matrix2d));
        Matrix::callAllocElementD(data, cfg.INPUT_SIZE_X* cfg.INPUT_SIZE_X, 1);
        dataBatch.push_back(data);
        Matrix::callAllocElementD(label, cfg.OUTPUT_SIZE,1);
        labelBatch.push_back(label);
    }
}

void DenseMLP::run() {

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

void DenseMLP::train() {
    int success=0;
    pastCost=0;
    for (int trial = 0; trial < cfg.TRAIN_BATCH_SIZE; trial++) {
        layers[0]->nodes = flattern(dataBatch[trial]);

        //forward feeding
        for (int i = 1; i < layers.size(); i++) {
            layers[i]->activate(layers[i - 1]);
        }

        //calculate cost
        costBuffer = *(*(*copyD2D(layers[layers.size()-1]->nodes, costBuffer) - labelBatch[trial])^2)*0.5;
        float cost = sumC(costBuffer);
        correctOut->nodes = labelBatch[trial];
        pastCost += cost;

        //calculate correction
        int maxIndex1 = 0, maxIndex2 = 0;
        Matrix::Matrix2d* debug;
        cudaMallocHost((void**)&debug, sizeof(Matrix::Matrix2d));
        Matrix::callAllocElementH(debug, DenseMLP::cfg.OUTPUT_SIZE, 1);
        cudaMemcpy(debug->elements, layers[3]->nodes->elements, sizeof(float) * cfg.OUTPUT_SIZE, cudaMemcpyDeviceToHost);
        for(int i=0; i< cfg.OUTPUT_SIZE; i++) {
            maxIndex1 = *(debug->elements + i) > *(debug->elements + maxIndex1) ? i : maxIndex1;
        }
        cudaMemcpy(debug->elements, correctOut->nodes->elements, sizeof(float) * cfg.OUTPUT_SIZE, cudaMemcpyDeviceToHost);
        for(int i=0; i< cfg.OUTPUT_SIZE; i++){
            maxIndex2 = *(debug->elements + i) > *(debug->elements + maxIndex2) ? i: maxIndex2;
        }
        success = maxIndex1 == maxIndex2 ? success+1 : success;

        //back propagate
        for (int i = (int)layers.size()-1; i > 0; i--) {
            layers[i]->propagate(layers[i - 1],i+1 < layers.size()? layers[i+1] : correctOut);
        }
        cudaFreeHost(debug->elements);
        cudaFreeHost(debug);
    }

    //apply changes (errors)
    for (int i = (int)layers.size()-1; i > 0; i--) {
        layers[i]->learn(cfg.TRAIN_BATCH_SIZE, cfg.LEARNING_RATE);
    }
    logInfo("Batch trained with cost: " + to_string(pastCost/(float)cfg.TRAIN_BATCH_SIZE) +
    " success rate: " + to_string(success));
}