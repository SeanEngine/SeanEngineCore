//
// Created by DanielSun on 12/28/2021.
//

#ifndef CUDANNGEN2_CONVLAYER_CUH
#define CUDANNGEN2_CONVLAYER_CUH

#include "Layer.cuh"
#include "../utils/NeuralUtils.cuh"
#include "../utils/logger.cuh"


class ConvLayer : public Layer{
public:
     Matrix::Matrix2d* filterBuffer, *featureMapBuffer, *zBuffer, *filterBiases, *filterBiasD, *filterDBuffer;
     Matrix::Matrix2d* featureMapTrans; //For back propagation
     Matrix::Tensor3d* paddedFeature, *z, *output;
     Matrix::Tensor4d* filters, *filterD;

    Matrix::Matrix2d* propaBuffer, *errors, *filterBufferTrans;
     int stride;

     string getType() override;

     ConvLayer(dim4 filterSize, dim3 featureMapSize, int stride, int layerID) :
          Layer((int)filterSize.z * (int)pow((featureMapSize.y-1)/stride+1,2)) {

         this->id = layerID;
         this->stride = stride;

         //for now this layer only accepts square filters and features, since the test on
         //allocating gemm with rectangular shaped features failed with a bug I can't fix
         assert(filterSize.y == filterSize.x && featureMapSize.y == featureMapSize.x);

         //calculate padding size
         dim3 paddedFeatureSize = dim3(featureMapSize.x+(filterSize.x/2)*2, featureMapSize.x+(filterSize.y/2)*2,
                                       featureMapSize.z);

         //register indexes
         cudaMallocHost(&filterBuffer, sizeof(Matrix::Matrix2d));     //empty
         cudaMallocHost(&zBuffer, sizeof(Matrix::Matrix2d));     //empty
         cudaMallocHost(&filterDBuffer, sizeof(Matrix::Matrix2d));

         //alloc all convolution volumes
         paddedFeature = Matrix::callAllocElementD(featureMapSize.z, paddedFeatureSize.y,paddedFeatureSize.x);
         filters = Matrix::callAllocElementD(filterSize.w, filterSize.z , filterSize.y, filterSize.x);
         filterD = Matrix::callAllocElementD(filterSize.w, filterSize.z , filterSize.y, filterSize.x);
         filterBiases = Matrix::callAllocElementD(filterSize.w, 1);
         filterBiasD = Matrix::callAllocElementD(filterSize.w, 1);
         z = Matrix::callAllocElementD(filterSize.w, (paddedFeatureSize.y-filterSize.y)/stride+1,
                                   (paddedFeatureSize.y-filterSize.y)/stride+1);
         output = Matrix::callAllocElementD(filterSize.w, (paddedFeatureSize.y-filterSize.y)/stride+1,
                                            (paddedFeatureSize.y-filterSize.y)/stride+1);

         //alloc buffers for gemm operation in convolution
         featureMapBuffer = Matrix::callAllocElementD(filterSize.x*filterSize.y*featureMapSize.z,
                                   z->rowcount * z->colcount);
         featureMapTrans = Matrix::callAllocElementD(featureMapBuffer->colcount, featureMapBuffer->rowcount);


         //this is used when we propagate the errors of this layer back to lower level layers
         propaBuffer = Matrix::callAllocElementD(filterSize.x * filterSize.y * featureMapSize.z,
                                           z->rowcount * z->colcount);

         //the error of this layer stored as 2d matrix for easier computation
         errors = Matrix::callAllocElementD(filterSize.w, ((paddedFeatureSize.y-filterSize.y)/stride+1) *
                                            (paddedFeatureSize.y-filterSize.y)/stride+1);

         //initialize buffers and alloc pointers to tensors they represents
         filterDBuffer->index(filterSize.w, filterSize.z * filterSize.y * filterSize.x, filterD->elements);

         this->nodes->elements = output->elements;

         logInfo("Layer register complete : " + to_string(id) + " " + getType() + " " + to_string(NODE_NUMBER));
         logInfo("CONV INFO: filters: " + filters->toString() + " padded features: " + paddedFeature->toString()
                                  + " z: " + z->toString());
     }

     //forward convolution operations
     void activate(Matrix::Tensor3d* prevFeatures);

     //back propagation
     void propagate(Matrix::Matrix2d* prevErrors, Matrix::Matrix2d *prevZBuffer);

     void recFilters() const;
     void recBiases() const;


     void activate(Layer *prevLayer) override;
     void propagate(Layer *prev, Layer *next) override;
     void learn(int BATCH_SIZE, float LEARNING_RATE) override;
};


#endif //CUDANNGEN2_CONVLAYER_CUH
