//
// Created by DanielSun on 12/28/2021.
//

#ifndef CUDANNGEN2_CONVLAYER_CUH
#define CUDANNGEN2_CONVLAYER_CUH

#include "Layer.cuh"
#include "../utils/logger.cuh"


class ConvLayer : public Layer{
public:
     Matrix::Matrix2d* filterBuffer, *featureMapBuffer, *outputBuffer;
     Matrix::Tensor3d* paddedFeature, *output;
     Matrix::Tensor4d* filters;
     string getType() override;

     ConvLayer(dim4 filterSize, dim3 featureMapSize, int stride, int layerID) :
          Layer((int)filterSize.z * (int)pow((featureMapSize.y-1)/stride+1,2)) {

         this->id = layerID;

         assert(filterSize.y == filterSize.x && featureMapSize.y == featureMapSize.x);
         assert(filterSize.y % 2 != 0);

         //calculate padding size
         dim3 paddedFeatureSize = dim3(featureMapSize.x+(filterSize.x/2)*2, featureMapSize.x+(filterSize.y/2)*2,
                                       featureMapSize.z);

         //register indexes
         cudaMallocHost(&filterBuffer, sizeof(Matrix::Matrix2d));     //empty
         cudaMallocHost(&outputBuffer, sizeof(Matrix::Matrix2d));     //empty

         //alloc all convolution volumes
         paddedFeature = Matrix::callAllocElementD(featureMapSize.z, paddedFeatureSize.y,paddedFeatureSize.x);
         filters = Matrix::callAllocElementD(filterSize.w, filterSize.z , filterSize.y, filterSize.x);
         output = Matrix::callAllocElementD(filterSize.z, (paddedFeatureSize.y-filterSize.y)/stride+1,
                                   (paddedFeatureSize.y-filterSize.y)/stride+1);

         //alloc buffers for gemm operation in convolution
         featureMapBuffer = Matrix::callAllocElementD(filterSize.x*filterSize.y*featureMapSize.z,
                                   output->rowcount * output->colcount);

         this->nodes->elements = output->elements;

         logInfo("Layer register complete : " + to_string(id) + " " + getType() + " " + to_string(NODE_NUMBER));
         logInfo("CONV INFO: filters: " + filters->toString() + " padded features: " + paddedFeature->toString()
                                  + " output: " + output->toString());
     }

     void activate(Layer *prevLayer) override;
     void propagate(Layer *prev, Layer *next) override;
     void learn(int BATCH_SIZE, float LEARNING_RATE) override;
};


#endif //CUDANNGEN2_CONVLAYER_CUH
