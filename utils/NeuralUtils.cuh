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


#endif //CUDANNGEN2_NEURALUTILS_CUH