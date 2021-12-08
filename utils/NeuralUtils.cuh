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
class NeuralUtils {
public:
    //activation methods
    static Matrix::Matrix2d* callActivationSigmoid(Matrix::Matrix2d* mat1);
    static Matrix::Matrix2d* callActivationSigmoid(Matrix::Matrix2d *mat1, Matrix::Matrix2d *result);
    static Matrix::Matrix2d* callDerivativeSigmoid(Matrix::Matrix2d *mat1, Matrix::Matrix2d *result);
    static Matrix::Matrix2d* callLeakyReluActivation(Matrix::Matrix2d *mat1, Matrix::Matrix2d *result, float ALPHA);
    static Matrix::Matrix2d* callLeakyReluDerivative(Matrix::Matrix2d *mat1, Matrix::Matrix2d *result, float ALPHA);
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
#endif //CUDANNGEN2_NEURALUTILS_CUH
