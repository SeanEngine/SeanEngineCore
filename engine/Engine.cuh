//
// Created by DanielSun on 11/9/2021.
//

#ifndef CUDANNGEN2_ENGINE_CUH
#define CUDANNGEN2_ENGINE_CUH


#include "../models/Model.cuh"

class Engine {
public:
    struct EngineConfig{
        //compute configs (default)
        int CUDA_DEVICE_INDEX = 0;
        int CPU_THREADS = 12;

        //execution configs  (default)
        const char* MODEL_LOAD_PATH;
        const char* MODEL_SAVE_PATH;
        const char* TRAIN_DATA_PATH;
        const char* TRAIN_LABEL_PATH;
    };

    EngineConfig config;
    Model* models;

    //start running the engine
    void boot(EngineConfig conf, Model* model);

};


#endif //CUDANNGEN2_ENGINE_CUH
