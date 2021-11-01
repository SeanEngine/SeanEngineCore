//
// Created by DanielSun on 11/1/2021.
//

#ifndef CUDANNGEN2_LAYER_CUH
#define CUDANNGEN2_LAYER_CUH
#include "../utils/Matrix.cuh"
#include "string"

using namespace std;

//this is just a abstract form of "Layer" used for managements
class Layer {
public:
     int id{};
     virtual string getType();
};


#endif //CUDANNGEN2_LAYER_CUH
