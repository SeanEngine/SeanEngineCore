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
class DenseLayer : Layer{
public:
     Matrix::Matrix2d* nodes, *weights, *biases, *z, *errors;
     Matrix::Matrix2d* weightBuffer, *prevActivationBuffer;
     Matrix::Matrix2d* weightDerivatives, *biasDerivatives;
     int NODE_NUMBER, PREV_NODE_NUMBER, NEXT_NODE_NUMBER;
     string getType() override;

     DenseLayer(int NODE_NUMBER, int PREV_NODE_NUMBER, int NEXT_NODE_NUMBER, int LayerID){
         this->NODE_NUMBER = NODE_NUMBER;
         this->PREV_NODE_NUMBER = PREV_NODE_NUMBER;
         this->NEXT_NODE_NUMBER = NODE_NUMBER;
         this->id = LayerID;

         //allocate matrix callers
         cudaMallocHost((void**)(&nodes), sizeof(Matrix::Matrix2d));
         cudaMallocHost((void**)(&weights), sizeof(Matrix::Matrix2d));
         cudaMallocHost((void**)(&biases), sizeof(Matrix::Matrix2d));
         cudaMallocHost((void**)(&z), sizeof(Matrix::Matrix2d));
         cudaMallocHost((void**)(&errors), sizeof(Matrix::Matrix2d));
         cudaMallocHost((void**)(&weightBuffer), sizeof(Matrix::Matrix2d));
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
     }
     //The corresponding math formulas are recorded in the cu file for these methods;
     //calculate activation
     void activate(Matrix::Matrix2d* prevNodes);

     //calculate the errors as output layer
     void reflectOutput(Matrix::Matrix2d* correctOut);

     //calculate the errors
     void reflect(Matrix::Matrix2d* nextWeights, Matrix::Matrix2d* nextBias, Matrix::Matrix2d *nextErrors);

     //update the memories of weight derivatives
     void recWeights(Matrix::Matrix2d* prevActivations);

     //update the memories of bias derivatives
     void recBias();

     //apply the changes:
     void applyWeights(int BATCH_SIZE, float LEARNING_RATE);
};


#endif //CUDANNGEN2_DENSELAYER_CUH
