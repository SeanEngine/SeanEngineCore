//
// Created by DanielSun on 11/22/2021.
//

#include "DenseMLP.cuh"
#include "../io/Reader.cuh"
#include "../utils/logger.cuh"
#include "../utils/Matrix.cuh"
#include "../layers/DenseLayer.cuh"

int readDataset(const string& path0, vector<Matrix::Matrix2d*>* data, vector<Matrix::Matrix2d*>* label,
                 DenseMLP::Config cfg, int labelIndex, int count) {
    intptr_t hFile = 0;
    struct _finddata_t fileInfo{};
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
        Matrix::callAllocElementH(matData, cfg.BMP_READ_DIM,cfg.BMP_READ_DIM);
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
        Matrix::callAllocElementD(matT, cfg.BMP_READ_DIM,cfg.BMP_READ_DIM);
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
     logInfo("===========< REGISTERING : DenseMLP >============",0x05);
     layers.push_back(new Layer(784));  //input layer
     layers.push_back(new DenseLayer(0, 16, 784, 16, 1));
     layers.push_back(new DenseLayer(0, 16, 16, 10, 2));
     layers.push_back(new DenseLayer(0, 10, 16, 10, 3));
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
}

void DenseMLP::run() {

}

void DenseMLP::loadData() {

}

void DenseMLP::train() {

}

void DenseMLP::unloadData() {

}