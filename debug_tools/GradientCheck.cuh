//
// Created by DanielSun on 12/2/2021.
//

#ifndef CUDANNGEN2_GRADIENTCHECK_CUH
#define CUDANNGEN2_GRADIENTCHECK_CUH
#include <utility>
#include <vector>
#include "../layers/Layer.cuh"
#include "../layers/DenseLayer.cuh"
#include "../utils/Matrix.cuh"

/**
 * this is a debug tool for checking the gradient decent for back propagation
 * these methods use a reasonably small value of epsilon to approx the gredient, and check if they do align with the
 * value calculated directly from derivatives
 */
class GradientCheck {
public:
    static float gradientCheck(vector<Layer*> layers, Matrix::Matrix2d* input, Layer* label);
};

static float gradCheck(vector<Layer*> layers, Matrix::Matrix2d* input, Layer* label){
    return GradientCheck::gradientCheck(std::move(layers), input, label);
}

#endif //CUDANNGEN2_GRADIENTCHECK_CUH
