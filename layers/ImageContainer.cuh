//
// Created by DanielSun on 1/30/2022.
//

#ifndef CUDANNGEN2_IMAGECONTAINER_CUH
#define CUDANNGEN2_IMAGECONTAINER_CUH


#include "Layer.cuh"

class ImageContainer : public Layer{
public:
    Matrix::Tensor3d* features;

    string getType() override;

    ImageContainer(dim3 size) : Layer(size.x * size.y * size.z) {
        cudaMallocHost(&features, sizeof(Matrix::Tensor3d));
        features->index(size.z,size.y,size.x);
    }
};


#endif //CUDANNGEN2_IMAGECONTAINER_CUH
