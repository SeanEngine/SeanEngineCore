//
// Created by DanielSun on 1/29/2022.
//

#include <random>
#include "VGG16.cuh"
#include "../utils/logger.cuh"
#include "../layers/ConvLayer.cuh"
#include "../layers/MaxPoolingLayer.cuh"
#include "../layers/DenseLayer.cuh"
#include "../layers/SoftmaxLayer.cuh"
#include "../layers/ImageContainer.cuh"

void VGG16::registerModel() {
    logInfo("===========< REGISTERING : VGG16 >============",0x05);
    layers.push_back(new ImageContainer(dim3(224,224,3)));
    layers.push_back(new ConvLayer(dim4(3,3,3,64),dim3(224,224,3),1,0));
    layers.push_back(new ConvLayer(dim4(3,3,64,64),dim3(224,224,64),1,1));
    layers.push_back(new MaxPoolingLayer(dim3(224,224,64),2));
    layers.push_back(new ConvLayer(dim4(3,3,64,128),dim3(112,112,64),1,3));
    layers.push_back(new ConvLayer(dim4(3,3,128,128),dim3(112,112,128),1,4));
    layers.push_back(new MaxPoolingLayer(dim3(112,112,128),2));
    layers.push_back(new ConvLayer(dim4(3,3,128,256), dim3(56,56,128), 1, 6));
    layers.push_back(new ConvLayer(dim4(3,3,256,256), dim3(56,56,256), 1, 7));
    layers.push_back(new MaxPoolingLayer(dim3(56,56,256),2));
    layers.push_back(new ConvLayer(dim4(3,3,256,512),dim3(28,28,256),1,9));
    layers.push_back(new ConvLayer(dim4(3,3,512,512),dim3(28,28,512),1,10));
    layers.push_back(new ConvLayer(dim4(3,3,512,512),dim3(28,28,512),1,11));
    layers.push_back(new MaxPoolingLayer(dim3(28,28,512),2));
    layers.push_back(new ConvLayer(dim4(3,3,512,512),dim3(14,14,512),1,13));
    layers.push_back(new ConvLayer(dim4(3,3,512,512),dim3(14,14,512),1,14));
    layers.push_back(new ConvLayer(dim4(3,3,512,512),dim3(14,14,512),1,15));
    layers.push_back(new MaxPoolingLayer(dim3(14,14,512),2));
    layers.push_back(new DenseLayer(4096,25088,4096,18));
    layers.push_back(new DenseLayer(4096,4096,1000,19));
    layers.push_back(new DenseLayer(1000, 4096, cfg.OUTPUT_SIZE,20));
    layers.push_back(new SoftmaxLayer(cfg.OUTPUT_SIZE,1000,cfg.OUTPUT_SIZE,21));
}

void VGG16::loadModel() {
     if(cfg.LOAD_MODEL_FROM_SAV){
        //load from save
         return;
     }
     for(Layer* layer : layers){
         layer->randomInit();
     }
}

void VGG16::loadDataSet() {
     vector<string> classes = Reader::getDirFiles(cfg.TRAIN_DATA_PATH);
     logInfo("Reading dataset classes : ", 0x06);
     int classIndex = 0;

     //get all classes
     for(const string& class0: classes){
         vector<string> imgNames = Reader::getDirFiles(class0);

         //read all images from each class
         for(int i=0; i<imgNames.size(); i++){
             auto* labelElement = Matrix::callAllocElementH(cfg.OUTPUT_SIZE,1);
             auto* dataElement = Matrix::callAllocElementH(cfg.DATA_SIZE.z, cfg.DATA_SIZE.y, cfg.DATA_SIZE.x);

             Matrix::callAllocZero(labelElement);
             labelElement->setH2D(classIndex,0,1.0F);

             dataset.push_back(dataElement);
             labelSet.push_back(labelElement);
         }
         Reader::readImageRGB(cfg.CPU_THREADS, cfg.DATA_SIZE, &imgNames, &dataset);
         logInfo("DATASET > read " + to_string(imgNames.size())+ " files for label : " + class0, 0x05);
         classIndex++;
     }

     //Alloc the batch space
     for(int i=0; i<cfg.TRAIN_BATCH_SIZE; i++){
         auto* data = Matrix::callAllocElementD(cfg.DATA_SIZE.z, cfg.DATA_SIZE.y, cfg.DATA_SIZE.x);
         dataBatch.push_back(data);
         auto* label = Matrix::callAllocElementD(cfg.OUTPUT_SIZE,1);
         labelBatch.push_back(label);
     }
}

void VGG16::loadData() {
    random_device rd;
    default_random_engine gen = default_random_engine(rd());
    uniform_int_distribution<int> dis(0,(int)dataset.size()-1);
    for(int i=0; i<cfg.TRAIN_BATCH_SIZE; i++){
        int index = dis(gen);
        copyH2D(dataset[index], dataBatch[i]);
        copyH2D(labelSet[index], labelBatch[i]);
    }
}

int VGG16::calcCorrection(int success, Layer* correct){
    //calculate correction
    int maxIndex1 = 0, maxIndex2 = 0;
    auto* debug = Matrix::callAllocElementH(cfg.OUTPUT_SIZE, 1);
    cudaMemcpy(debug->elements, layers[3]->nodes->elements, sizeof(float) * cfg.OUTPUT_SIZE, cudaMemcpyDeviceToHost);
    for(int i=0; i< cfg.OUTPUT_SIZE; i++) {
        maxIndex1 = *(debug->elements + i) > *(debug->elements + maxIndex1) ? i : maxIndex1;
    }
    cudaMemcpy(debug->elements, correct->nodes->elements, sizeof(float) * cfg.OUTPUT_SIZE, cudaMemcpyDeviceToHost);
    for(int i=0; i< cfg.OUTPUT_SIZE; i++){
        maxIndex2 = *(debug->elements + i) > *(debug->elements + maxIndex2) ? i: maxIndex2;
    }
    cudaFreeHost(debug->elements);
    cudaFreeHost(debug);
    return maxIndex1 == maxIndex2 ? success+1 : success;
}

void VGG16::train() {
    int success=0;
    float pastCost=0;
    auto* correct = new Layer(10);
    auto* costBuffer = Matrix::callAllocElementD(cfg.OUTPUT_SIZE, 1);
    for (int trial = 0; trial < cfg.TRAIN_BATCH_SIZE; trial++) {
        ((ImageContainer*)layers[0])->features->elements = dataBatch[trial]->elements;
        for (int i=1; i<layers.size(); i++){
            layers[i]->activate(layers[i-1]);
        }
        
        correct->nodes = labelBatch[0];

        //calc costs
        costBuffer = softMaxL(layers[layers.size()-1]->nodes, labelBatch[trial], costBuffer);
        float cost = sumH(costBuffer);
        pastCost+=cost;

        //calc success rate
        success = calcCorrection(success, correct);
        //propagate
        for (int i=(int)layers.size()-1; i>1; i--){
            layers[i]->propagate(layers[i-1], i+1 > 21 ? correct : layers[i+1]);
        }
    }
    for (int i=1; i<layers.size(); i++) {
        layers[i]->learn(cfg.TRAIN_BATCH_SIZE, cfg.LEARNING_RATE);
    }

    logInfo("Batch trained with cost: " + to_string(pastCost/(float)cfg.TRAIN_BATCH_SIZE) +
            " success rate: " + to_string((float)success/(float)cfg.TRAIN_BATCH_SIZE));

}

void VGG16::run() {

}
