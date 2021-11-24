//
// Created by DanielSun on 11/9/2021.
//

#ifndef CUDANNGEN2_MODEL_CUH
#define CUDANNGEN2_MODEL_CUH
#include "../engine/Engine.cuh"

class Model {
public:
    //register model
    virtual void registerModel() = 0;

    //load model
    virtual void loadModel() = 0;

    //load dataset from SSD
    virtual void loadDataSet() = 0;

    //load the data batch from dataset (host memory to device memory)
    virtual void loadData() = 0;

    //train a batch on the model
    virtual void train() = 0;

    //run the model
    virtual void run() = 0;
};


#endif //CUDANNGEN2_MODEL_CUH
