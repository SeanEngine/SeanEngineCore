//
// Created by DanielSun on 11/1/2021.
//

#ifndef CUDANNGEN2_DENSELAYER_CUH
#define CUDANNGEN2_DENSELAYER_CUH
#include "../utils/Matrix.cuh"
#include "../utils/logger.cuh"
#include "Layer.cuh"
#include "string"

using namespace std;
class DenseLayer : public Layer{
public:
     Matrix::Matrix2d* weights{}, *biases{}, *z{}, *errors{};
     Matrix::Matrix2d* weightBuffer{}, *prevActivationBuffer{};
     Matrix::Matrix2d* weightDerivatives{}, *biasDerivatives{};
     int NODE_NUMBER, PREV_NODE_NUMBER, NEXT_NODE_NUMBER;
     float WEIGHT_DECAY = 1.0f;
     string getType() override;

     DenseLayer( int NODE_NUMBER, int PREV_NODE_NUMBER, int NEXT_NODE_NUMBER, int LayerID)
             : Layer(NODE_NUMBER) {
         this->NODE_NUMBER = NODE_NUMBER;
         this->PREV_NODE_NUMBER = PREV_NODE_NUMBER;
         this->NEXT_NODE_NUMBER = NEXT_NODE_NUMBER;
         this->id = LayerID;

         //allocate matrix callers
         cudaMallocHost((void**)(&nodes), sizeof(Matrix::Matrix2d));
         cudaMallocHost((void**)(&weights), sizeof(Matrix::Matrix2d));
         cudaMallocHost((void**)(&biases), sizeof(Matrix::Matrix2d));
         cudaMallocHost((void**)(&z), sizeof(Matrix::Matrix2d));
         cudaMallocHost((void**)(&errors), sizeof(Matrix::Matrix2d));
         cudaMallocHost((void**)(&weightBuffer), sizeof(Matrix::Matrix2d)); // for transpose operation
         cudaMallocHost((void**)(&prevActivationBuffer), sizeof(Matrix::Matrix2d));
         cudaMallocHost((void**)(&weightDerivatives), sizeof(Matrix::Matrix2d));
         cudaMallocHost((void**)(&biasDerivatives), sizeof(Matrix::Matrix2d));

         //allocate matrix
         Matrix::callAllocElementD(nodes, NODE_NUMBER, 1);
         Matrix::callAllocElementD(weights, NODE_NUMBER, PREV_NODE_NUMBER);
         Matrix::callAllocElementD(biases, NODE_NUMBER, 1);
         Matrix::callAllocElementD(z, NODE_NUMBER, 1);
         Matrix::callAllocElementD(weightDerivatives, NODE_NUMBER, PREV_NODE_NUMBER);
         Matrix::callAllocElementD(errors, NODE_NUMBER, 1);
         Matrix::callAllocElementD(weightBuffer, NODE_NUMBER, NEXT_NODE_NUMBER);
         Matrix::callAllocElementD(prevActivationBuffer, 1, PREV_NODE_NUMBER);
         Matrix::callAllocElementD(biasDerivatives, NODE_NUMBER, 1);

         logInfo("Layer register complete : " + to_string(id) + " DENSE " + to_string(NODE_NUMBER));
         logInfo("cuda memory occupation :" + to_string((double)(sizeof(float)*(5*NODE_NUMBER + 2*NODE_NUMBER*PREV_NODE_NUMBER
         + NODE_NUMBER * NEXT_NODE_NUMBER))/(1024*1024)));
     }
     //The corresponding math formulas are recorded in the cu file for these methods;
     //calculate activation
     void calcActivate(Matrix::Matrix2d* prevNodes);

     //calculate the errors as output layer
     void propagatingOutput(Matrix::Matrix2d* correctOut);

     //calculate the errors
     void propagate(Matrix::Matrix2d* nextWeights, Matrix::Matrix2d *nextErrors);

     //update the memories of weight derivatives
     void recWeights(Matrix::Matrix2d* prevActivations);

     //update the memories of bias derivatives
     void recBias();

     //apply the changes:
     void applyWeights(int BATCH_SIZE, float LEARNING_RATE);
     void applyBias(int BATCH_SIZE, float LEARNING_RATE);

     //overriding methods for calling:
     void activate(Layer *prev) override;
     void propagate(Layer *prev, Layer *next) override;
     void learn(int BATCH_SIZE, float LEARNING_RATE) override;
};


#endif //CUDANNGEN2_DENSELAYER_CUH
