//
// Created by DanielSun on 11/9/2021.
//

#ifndef CUDANNGEN2_ENGINE_CUH
#define CUDANNGEN2_ENGINE_CUH


#include "../models/Model.cuh"
struct EngineConfig{
    //compute configs (default)
     int CUDA_DEVICE_INDEX = 0;
     int CPU_THREADS = 12;

    //execution configs  (default)
     char* MODEL_LOAD_PATH;
     char* MODEL_SAVE_PATH;
     char* TRAIN_DATA_PATH = "C:\\Users\\DanielSun\\Desktop\\resources\\MLDatasets\\convproc\\train";
     char* TRAIN_LABEL_PATH;
     char* TEST_DATA_PATH;
     char* TEST_LABEL_PATH;

     int TRAIN_BATCH_SIZE = 100;
     float LEARNING_RATE = 0.03f;
     bool LOAD_MODEL_FROM_SAV = false;
};

class Engine {
public:

    static EngineConfig config;
    //Model* models;

    //start running the engine
    //void boot(EngineConfig conf, Model* model);

};


#endif //CUDANNGEN2_ENGINE_CUH
