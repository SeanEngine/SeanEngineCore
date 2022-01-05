//
// Created by DanielSun on 12/8/2021.
//

#ifndef CUDANNGEN2_NEURALUTILS_CUH
#define CUDANNGEN2_NEURALUTILS_CUH
#include "Matrix.cuh"

/**
 * this class is for all the other utilities that are not 
 * considered to be a part of the matrix operations
 */
static const float RELU_ALPHA = 0.01f;
static const int CUDA_SOFTMAX_BLOCK = 1024;
static const int CONV_SIZE = 3;
class NeuralUtils {
public:
    //activation methods
    static Matrix::Matrix2d* callActivationSigmoid(Matrix::Matrix2d* mat1);
    static Matrix::Matrix2d* callActivationSigmoid(Matrix::Matrix2d *mat1, Matrix::Matrix2d *result);
    static Matrix::Matrix2d* callDerivativeSigmoid(Matrix::Matrix2d *mat1, Matrix::Matrix2d *result);
    static Matrix::Matrix2d* callLeakyReluActivation(Matrix::Matrix2d *mat1, Matrix::Matrix2d *result, float ALPHA);
    static Matrix::Matrix2d* callLeakyReluDerivative(Matrix::Matrix2d *mat1, Matrix::Matrix2d *result, float ALPHA);

    //loss
    static Matrix::Matrix2d* callSoftMax(Matrix::Matrix2d* mat1, Matrix::Matrix2d* result,  float* buffer);
    static Matrix::Matrix2d* callSoftMaxDerivatives(Matrix::Matrix2d* mat1, Matrix::Matrix2d* correctOut, Matrix::Matrix2d* result);
    static Matrix::Matrix2d* callSoftMaxCost(Matrix::Matrix2d* mat1,Matrix::Matrix2d *correctOut, Matrix::Matrix2d* result);

    //conv
    /**
     * @see{https://sahnimanas.github.io/post/anatomy-of-a-high-performance-convolution/}
     * This method is for convolution with 4D filters and 3D feature maps
     * This method only support odd filter size (1x1, 3x3, 5x5,....)
     * @param mat1  The import feature map with dimensions D * H0 * W0
     * @param filter The 4D filter with dimensions N * D * K * K
     * @param result The output with dimensions N * H * W
     * @param stride The stride for convolution (steps of the filters)
     * @param filterBuffer an empty initialized Matrix2d object
     * @param featureBuffer The 2d buffer that stores processed features with Dimensions K^2 * D , H * W
     * @param outputBuffer an empty initialized Matrix2d object
     * @return convoluted features
     */
    static Matrix::Tensor3d* callConv2d(Matrix::Tensor3d *mat1, Matrix::Tensor4d *filter, Matrix::Tensor3d *result, unsigned int stride,
                                        Matrix::Matrix2d* filterBuffer, Matrix::Matrix2d* featureBuffer, Matrix::Matrix2d* outputBuffer);
    static Matrix::Tensor3d* padding3d(Matrix::Tensor3d *mat1, Matrix::Tensor3d *output, unsigned int padSize);
};

static Matrix::Matrix2d* sigmoid(Matrix::Matrix2d* mat1, Matrix::Matrix2d* result){
    return NeuralUtils::callActivationSigmoid(mat1, result);
}

static Matrix::Matrix2d* sigmoidD(Matrix::Matrix2d* mat1, Matrix::Matrix2d* result){
    return NeuralUtils::callDerivativeSigmoid(mat1, result);
}

static Matrix::Matrix2d* lRelu(Matrix::Matrix2d* mat1, Matrix::Matrix2d* result){
    return NeuralUtils::callLeakyReluActivation(mat1, result, RELU_ALPHA);
}

static Matrix::Matrix2d* lReluD(Matrix::Matrix2d* mat1, Matrix::Matrix2d* result){
    return NeuralUtils::callLeakyReluDerivative(mat1, result, RELU_ALPHA);
}

static Matrix::Matrix2d* softmax(Matrix::Matrix2d* mat1, Matrix::Matrix2d* result, float* buffer){
    return NeuralUtils::callSoftMax(mat1, result, buffer);
}

static Matrix::Matrix2d* softmaxD(Matrix::Matrix2d* mat1, Matrix::Matrix2d* correctOut, Matrix::Matrix2d* result){
    return NeuralUtils::callSoftMaxDerivatives(mat1, correctOut, result);
}
//calculate the loss of the softmax output
static Matrix::Matrix2d* softMaxL(Matrix::Matrix2d* mat1, Matrix::Matrix2d* correctOut, Matrix::Matrix2d* result){
    return NeuralUtils::callSoftMaxCost(mat1, correctOut, result);
}

static Matrix::Tensor3d* conv2D(Matrix::Tensor3d *mat1, Matrix::Tensor4d *filter, Matrix::Tensor3d *result, unsigned int stride,
                                Matrix::Matrix2d* filterBuffer, Matrix::Matrix2d* featureBuffer, Matrix::Matrix2d* outputBuffer){
    return NeuralUtils::callConv2d(mat1,filter, result, stride, filterBuffer, featureBuffer, outputBuffer);
}


#endif //CUDANNGEN2_NEURALUTILS_CUH
