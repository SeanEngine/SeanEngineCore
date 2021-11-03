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
     Matrix::Matrix2d* nodes, *weights, *biases, *z, *weightDerivatives, *errors;
     int NODE_NUMBER, PREV_NODE_NUMBER;
     string getType() override;

     DenseLayer(int NODE_NUMBER, int PREV_NODE_NUMBER, int LayerID){
         this->NODE_NUMBER = NODE_NUMBER;
         this->PREV_NODE_NUMBER = PREV_NODE_NUMBER;
         this->id = LayerID;

         //allocate matrix callers
         cudaMallocHost((void**)(&nodes), sizeof(Matrix::Matrix2d));
         cudaMallocHost((void**)(&weights), sizeof(Matrix::Matrix2d));
         cudaMallocHost((void**)(&biases), sizeof(Matrix::Matrix2d));
         cudaMallocHost((void**)(&z), sizeof(Matrix::Matrix2d));
         cudaMallocHost((void**)(&weightDerivatives), sizeof(Matrix::Matrix2d));
         cudaMallocHost((void**)(&errors), sizeof(Matrix::Matrix2d));

         //allocate matrix
         Matrix::callAllocElementD(nodes, NODE_NUMBER, 1);
         Matrix::callAllocElementD(weights, NODE_NUMBER, PREV_NODE_NUMBER);
         Matrix::callAllocElementD(biases, NODE_NUMBER, 1);
         Matrix::callAllocElementD(z, NODE_NUMBER, 1);
         Matrix::callAllocElementD(weightDerivatives, NODE_NUMBER, PREV_NODE_NUMBER);
         Matrix::callAllocElementD(errors, NODE_NUMBER, 1);

         logInfo("Layer construction complete : " + to_string(id) + " DENSE " + to_string(NODE_NUMBER));
     }

     void activate(Matrix::Matrix2d* prevNodes);

     void reflectOutput(Matrix::Matrix2d* correctOut);


};


#endif //CUDANNGEN2_DENSELAYER_CUH
