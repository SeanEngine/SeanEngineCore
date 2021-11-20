//
// Created by DanielSun on 11/1/2021.
//

#ifndef CUDANNGEN2_LAYER_CUH
#define CUDANNGEN2_LAYER_CUH
#include "../utils/Matrix.cuh"
#include "string"

using namespace std;

//this is a container layer without any actual functionality
class Layer {
public:
     int id{};
     virtual string getType();
     Matrix::Matrix2d *nodes{};

     //calculate activation
     virtual void activate(Layer* prevLayer);

     //propagate
     virtual void propagate(Layer* prev, Layer* next) = 0;

     //apply
     virtual void learn(int BATCH_SIZE, float LEARNING_RATE) = 0;

};


#endif //CUDANNGEN2_LAYER_CUH
