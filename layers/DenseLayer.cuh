//
// Created by DanielSun on 11/1/2021.
//

#ifndef CUDANNGEN2_DENSELAYER_CUH
#define CUDANNGEN2_DENSELAYER_CUH
#include "../utils/Matrix.cuh"
#include "../utils/NeuralUtils.cuh"
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

         //allocate matrix
         nodes = Matrix::callAllocElementD(NODE_NUMBER, 1);
         weights = Matrix::callAllocElementD(NODE_NUMBER, PREV_NODE_NUMBER);
         biases = Matrix::callAllocElementD(NODE_NUMBER, 1);
         z =Matrix::callAllocElementD( NODE_NUMBER, 1);
         weightDerivatives = Matrix::callAllocElementD(NODE_NUMBER, PREV_NODE_NUMBER);
         errors = Matrix::callAllocElementD(NODE_NUMBER, 1);
         weightBuffer = Matrix::callAllocElementD(NODE_NUMBER, NEXT_NODE_NUMBER);
         prevActivationBuffer = Matrix::callAllocElementD(1, PREV_NODE_NUMBER);
         biasDerivatives = Matrix::callAllocElementD( NODE_NUMBER, 1);

         logInfo("Layer register complete : " + to_string(id) + " " + getType() + " " + to_string(NODE_NUMBER));
         logInfo("cuda memory occupation :" + to_string((double)(sizeof(float)*(5*NODE_NUMBER + 2*NODE_NUMBER*PREV_NODE_NUMBER
         + NODE_NUMBER * NEXT_NODE_NUMBER))/(1024*1024)));
     }
     //The corresponding math formulas are recorded in the cu file for these methods;
     //these methods are overrided in the softmax layer so they are defined as virtual
     //calculate activation
     virtual void calcActivate(Matrix::Matrix2d* prevNodes);

     //calculate the errors as output layer
     virtual void propagatingOutput(Matrix::Matrix2d* correctOut);

     //calculate the errors
     virtual void propagate(Matrix::Matrix2d* nextWeights, Matrix::Matrix2d *nextErrors);

     //update the memories of weight derivatives
     void recWeights(Matrix::Matrix2d* prevActivations);

     //update the memories of bias derivatives
     void recBias();

     float calcQuadraticCost(Matrix::Matrix2d *correctOut);

     //apply the changes:
     void applyWeights(int BATCH_SIZE, float LEARNING_RATE);
     void applyBias(int BATCH_SIZE, float LEARNING_RATE);

     //overriding methods for calling:
     void activate(Layer *prev) override;
     void propagate(Layer *prev, Layer *next) override;
     void learn(int BATCH_SIZE, float LEARNING_RATE) override;
};


#endif //CUDANNGEN2_DENSELAYER_CUH
