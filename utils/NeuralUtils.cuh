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
    static Matrix::Matrix2d *callActivationSigmoid(Matrix::Matrix2d *mat1);

    static Matrix::Matrix2d *callActivationSigmoid(Matrix::Matrix2d *mat1, Matrix::Matrix2d *result);

    static Matrix::Matrix2d *callDerivativeSigmoid(Matrix::Matrix2d *mat1, Matrix::Matrix2d *result);

    static Matrix::Matrix2d *callActivationLeakyRelu(Matrix::Matrix2d *mat1, Matrix::Matrix2d *result, float ALPHA);

    static Matrix::Matrix2d *callDerivativeLeakyRelu(Matrix::Matrix2d *mat1, Matrix::Matrix2d *result, float ALPHA);

    static Matrix::Tensor *callActivationLeakyRelu(Matrix::Tensor *mat1, Matrix::Tensor *result, float ALPHA);

    static Matrix::Tensor *callDerivativeLeakyRelu(Matrix::Tensor *mat1, Matrix::Tensor *result, float ALPHA);

    //loss
    static Matrix::Matrix2d *callSoftMax(Matrix::Matrix2d *mat1, Matrix::Matrix2d *result, float *buffer);

    static Matrix::Matrix2d *
    callSoftMaxDerivatives(Matrix::Matrix2d *mat1, Matrix::Matrix2d *correctOut, Matrix::Matrix2d *result);

    static Matrix::Matrix2d *
    callSoftMaxCost(Matrix::Matrix2d *mat1, Matrix::Matrix2d *correctOut, Matrix::Matrix2d *result);

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
    static Matrix::Tensor3d *
    callConv2d(Matrix::Tensor3d *mat1, Matrix::Tensor4d *filter, Matrix::Tensor3d *result, unsigned int stride,
               Matrix::Matrix2d *filterBuffer, Matrix::Matrix2d *featureBuffer, Matrix::Matrix2d *outputBuffer,
               Matrix::Matrix2d *biases);

    /**
     * This method will pad zeros around the feature maps
     * @param mat1 input
     * @param output output (size = input + 2*padSize)
     * @param padSize tge size of the padding
     * @return padded features
     */
    static Matrix::Tensor3d *padding3d(Matrix::Tensor3d *mat1, Matrix::Tensor3d *output, unsigned int padSize);

    /**
     *  Allocate the biases for the convolution gemm operation
     * @param outputBuffer the buffer of output, size = N, H*W
     * @param bias the array for biases of each 3d conv kernels, size = N, 1
     * @return the output buffer that are pre-filled with the biases
     */
    static Matrix::Matrix2d *callAllocConvBias(Matrix::Matrix2d *outputBuffer, Matrix::Matrix2d *bias);

    /**
     * Img2Col operation rearranges the 3D feature maps into a 2d matrix for gemm convolution
     * @param featureMaps (size = C, H0, W0)
     * @param featureBuffer (size = K^2*C, H*W)
     * @param outputHeight H
     * @param filterSize K
     * @param stride
     * @return
     */
    static Matrix::Matrix2d *
    callImg2Col(Matrix::Tensor3d *featureMaps, Matrix::Matrix2d *featureBuffer, unsigned int outputHeight,
                unsigned int filterSize, unsigned int stride);

    static Matrix::Matrix2d *
    callCol2Img2dNP(Matrix::Matrix2d *errors, Matrix::Matrix2d *propaBuffer, unsigned int outputHeight,
                    unsigned int sourceHeight, unsigned int filterSize, unsigned int stride, unsigned int padSize);

    static Matrix::Matrix2d *callRowReduce(Matrix::Matrix2d *mat1, Matrix::Matrix2d *output, float *buffer);

    static Matrix::Tensor3d *
    callMaxPooling(Matrix::Tensor3d *source, Matrix::Tensor3d *record, Matrix::Tensor3d *output, int stride);

    static Matrix::Tensor3d *callMaxPooling(Matrix::Tensor3d *source, Matrix::Tensor3d *output, int stride);

    static Matrix::Tensor3d *
    callInvertMaxPool(Matrix::Tensor3d *errors, Matrix::Tensor3d *record, Matrix::Tensor3d *prevErrors, int stride);
};

static Matrix::Matrix2d *rowReduce(Matrix::Matrix2d *mat1, Matrix::Matrix2d *output, float *buffer) {
    return NeuralUtils::callRowReduce(mat1, output, buffer);
}

static Matrix::Matrix2d *sigmoid(Matrix::Matrix2d *mat1, Matrix::Matrix2d *result) {
    return NeuralUtils::callActivationSigmoid(mat1, result);
}

static Matrix::Matrix2d *sigmoidD(Matrix::Matrix2d *mat1, Matrix::Matrix2d *result) {
    return NeuralUtils::callDerivativeSigmoid(mat1, result);
}

static Matrix::Matrix2d *lRelu(Matrix::Matrix2d *mat1, Matrix::Matrix2d *result) {
    return NeuralUtils::callActivationLeakyRelu(mat1, result, RELU_ALPHA);
}

static Matrix::Matrix2d *lReluD(Matrix::Matrix2d *mat1, Matrix::Matrix2d *result) {
    return NeuralUtils::callDerivativeLeakyRelu(mat1, result, RELU_ALPHA);
}

static Matrix::Tensor *lRelu(Matrix::Tensor *mat1, Matrix::Tensor *result) {
    return NeuralUtils::callActivationLeakyRelu(mat1, result, RELU_ALPHA);
}

static Matrix::Matrix2d *softmax(Matrix::Matrix2d *mat1, Matrix::Matrix2d *result, float *buffer) {
    return NeuralUtils::callSoftMax(mat1, result, buffer);
}

static Matrix::Matrix2d *softmaxD(Matrix::Matrix2d *mat1, Matrix::Matrix2d *correctOut, Matrix::Matrix2d *result) {
    return NeuralUtils::callSoftMaxDerivatives(mat1, correctOut, result);
}

//calculate the loss of the softmax output
static Matrix::Matrix2d *softMaxL(Matrix::Matrix2d *mat1, Matrix::Matrix2d *correctOut, Matrix::Matrix2d *result) {
    return NeuralUtils::callSoftMaxCost(mat1, correctOut, result);
}

static Matrix::Tensor3d *
conv2D(Matrix::Tensor3d *mat1, Matrix::Tensor4d *filter, Matrix::Tensor3d *result, unsigned int stride,
       Matrix::Matrix2d *filterBuffer, Matrix::Matrix2d *featureBuffer, Matrix::Matrix2d *outputBuffer,
       Matrix::Matrix2d *biases) {
    return NeuralUtils::callConv2d(mat1, filter, result, stride, filterBuffer, featureBuffer, outputBuffer, biases);
}

static Matrix::Tensor3d *pad3D(Matrix::Tensor3d *mat1, Matrix::Tensor3d *output, unsigned int padSize) {
    return NeuralUtils::padding3d(mat1, output, padSize);
}

static Matrix::Matrix2d *
img2col(Matrix::Tensor3d *featureMaps, Matrix::Matrix2d *featureBuffer, unsigned int outputHeight,
        unsigned int filterSize, unsigned int stride) {
    return NeuralUtils::callImg2Col(featureMaps, featureBuffer, outputHeight, filterSize, stride);
}

static Matrix::Matrix2d *col2img2D(Matrix::Matrix2d *errors, Matrix::Matrix2d *propaBuffer, unsigned int outputHeight,
                                   unsigned int sourceHeight, unsigned int filterSize, unsigned int stride,
                                   unsigned int padSize) {
    return NeuralUtils::callCol2Img2dNP(errors, propaBuffer, outputHeight, sourceHeight, filterSize, stride, padSize);
}

#endif //CUDANNGEN2_NEURALUTILS_CUH
