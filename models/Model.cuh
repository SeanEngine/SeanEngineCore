//
// Created by DanielSun on 11/9/2021.
//

#ifndef CUDANNGEN2_MODEL_CUH
#define CUDANNGEN2_MODEL_CUH

class Model {
public:
    //load model
    virtual void loadModel();

    //load dataset from SSD
    virtual void loadDataSet();

    //load the data batch from dataset (host memory to device memory)
    virtual void loadData();

    //unload the data
    virtual void unloadData();

    //train a batch on the model
    virtual void train();

    //run the model
    virtual void execute();
};


#endif //CUDANNGEN2_MODEL_CUH
