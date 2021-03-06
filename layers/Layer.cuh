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
    Layer(int NODE_NUMBER) {
        this->NODE_NUMBER = NODE_NUMBER;
    }

    int id;
    unsigned int NODE_NUMBER;
    Layer* prevLayer, *nextLayer;

    virtual string getType();

    Matrix::Matrix2d *nodes{};

    Layer* bindPrev(Layer* prev);

    Layer* bindNext(Layer* next);

    Layer* bind(Layer* prev, Layer* next);

    //calculate activation
    virtual void activate(Layer *prevLayer);

    //propagate
    virtual void propagate(Layer *prev, Layer *next);

    //apply
    virtual void learn(int BATCH_SIZE, float LEARNING_RATE);

    virtual void randomInit();

    //use bind params
    void activate();

    void propagate();

};


#endif //CUDANNGEN2_LAYER_CUH
