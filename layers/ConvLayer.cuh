//
// Created by DanielSun on 12/28/2021.
//

#ifndef CUDANNGEN2_CONVLAYER_CUH
#define CUDANNGEN2_CONVLAYER_CUH

#include "Layer.cuh"
#include "../utils/logger.cuh"

class ConvLayer : public Layer{
public:
     Matrix::Matrix2d* filterBuffer, *featureMapBuffer;
     Matrix::Matrix3d* paddedFeature, *filters, *output;
     string getType() override;

     ConvLayer(dim3 filterSize, dim3 featureMapSize, int stride, int layerID) :
          Layer((int)filterSize.z * (int)pow((featureMapSize.y-1)/stride+1,2)) {

         this->id = layerID;

         assert(filterSize.y == filterSize.x && featureMapSize.y == featureMapSize.x);
         assert(filterSize.y % 2 != 0);

         //calculate padding size
         dim3 paddedFeatureSize = dim3(featureMapSize.x+(filterSize.x/2)*2, featureMapSize.x+(filterSize.y/2)*2,
                                       featureMapSize.z);

         //register indexes
         cudaMallocHost(&paddedFeature, sizeof(Matrix::Matrix3d));
         cudaMallocHost(&filters, sizeof(Matrix::Matrix3d));
         cudaMallocHost(&output, sizeof(Matrix::Matrix3d));
         cudaMallocHost(&filterBuffer, sizeof(Matrix::Matrix2d));
         cudaMallocHost(&featureMapBuffer, sizeof(Matrix::Matrix2d));

         //alloc all convolution volumes
         Matrix::callAllocElementD(paddedFeature, featureMapSize.z, paddedFeatureSize.y,paddedFeatureSize.x);
         Matrix::callAllocElementD(filters, filterSize.z, filterSize.y, filterSize.x);
         Matrix::callAllocElementD(output, filterSize.z, (paddedFeatureSize.y-filterSize.y)/stride+1,
                                   (paddedFeatureSize.y-filterSize.y)/stride+1);

         //alloc buffers for gemm operation in convolution
         Matrix::callAllocElementD(filterBuffer, filterSize.z, filterSize.x*filterSize.y*featureMapSize.z);
         Matrix::callAllocElementD(featureMapBuffer,filterSize.x*filterSize.y*featureMapSize.z,
                                   output->rowcount * output->colcount);

         logInfo("Layer register complete : " + to_string(id) + " " + getType() + " " + to_string(NODE_NUMBER));
         logInfo("CONV INFO: filters: " + filters->toString() + " padded features: " + paddedFeature->toString()
                                  + " output: " + output->toString());
     }

     void activate(Layer *prevLayer) override;
     void propagate(Layer *prev, Layer *next) override;
     void learn(int BATCH_SIZE, float LEARNING_RATE) override;
};


#endif //CUDANNGEN2_CONVLAYER_CUH
