//
// Created by DanielSun on 12/2/2021.
//

#include "../utils/logger.cuh"
#include "GradientCheck.cuh"
float GradientCheck::gradientCheck(vector<Layer*> layers, Matrix::Matrix2d* input, Layer* label) {
    layers[0]->nodes = input;
    float EPSILON = 1e-4;
    int flattenSize = 0;
    double divergence;

    //register the flat vector
    for(int i=1; i<layers.size(); i++){
        flattenSize += ((DenseLayer*)layers[i])->weights->rowcount * ((DenseLayer*)layers[i])->weights->colcount;
        flattenSize += ((DenseLayer*)layers[i])->biases->rowcount;
    }

    // row 0: all parameter values, row 1: all approximate slopes, row 2: all propagate slopes
    Matrix::Matrix2d* theta;
    cudaMallocHost((void**)&theta, sizeof(Matrix::Matrix2d));
    Matrix::callAllocElementH(theta, 3, flattenSize);

    logInfo("CHECK > starting gradient check with parameters: " + to_string(flattenSize));

    //activate the model to get the gradients
    for (int i = 1; i < layers.size(); i++) layers[i]->activate(layers[i - 1]);
    for (int i = (int)layers.size()-1; i > 0; i--) {
        layers[i]->propagate(layers[i - 1], i + 1 < layers.size() ? layers[i + 1] : label);
    }

    //construct processing array
    int fillIndex = 0;
    vector<int> sections;
    sections.push_back(0);
    for(int i=1; i<layers.size(); i++){
        auto* temp = (DenseLayer*)layers[i];

        //copy parameter values to row 0
        cudaMemcpy(theta->elements + fillIndex, temp->weights->elements, sizeof(float)* temp->weights->rowcount
        * temp->weights->colcount, cudaMemcpyDeviceToHost);

        //copy gradient to row 2
        cudaMemcpy(2*flattenSize + theta->elements + fillIndex, temp->weightDerivatives->elements, sizeof(float)*
        temp->weights->rowcount * temp->weights->colcount, cudaMemcpyDeviceToHost);
        fillIndex += temp->weights->rowcount * temp->weights->colcount;
        sections.push_back(fillIndex);
        logInfo("CHECK > parameter copy : weights : filled index = " + to_string(fillIndex));

        //do the same thing for biases
        cudaMemcpy(theta->elements + fillIndex, temp->biases->elements, sizeof(float)* temp->biases->colcount
        , cudaMemcpyDeviceToHost);
        cudaMemcpy(theta->elements + fillIndex + 2*flattenSize, temp->biasDerivatives->elements, sizeof(float)*
        temp->biases->colcount, cudaMemcpyDeviceToHost);
        fillIndex += temp->biases->rowcount;
        sections.push_back(fillIndex);
        logInfo("CHECK > parameter copy : biases : filled index = " + to_string(fillIndex));
    }
    logInfo("CHECK > parameter copy complete");
    //Matrix::inspect(theta);
    int sectionIndex=0;
    for(int i = 0; i < flattenSize; i++){

        //set the modifier index correctly
        for (int j = 0; j < sections.size(); j++){
            if(sections[j] > i){
                sectionIndex = j-1;
            }
        }

        Matrix::Matrix2d* toModify = !sectionIndex%2 ?((DenseLayer*)layers[1+sectionIndex/2])->weights :
            ((DenseLayer*)layers[1+sectionIndex/2])->biases;

        //calc C(W) + epsilon
        toModify->setH2D(i-sections[sectionIndex], theta->elements[i] + EPSILON);
        for (int prop = 1; prop < layers.size(); prop++) layers[prop]->activate(layers[prop - 1]);
        float cost0 = ((DenseLayer*)(layers[layers.size()-1]))->calcQuadraticCost(label->nodes);

        //calc C(W) - epsilon
        toModify->setH2D(i-sections[sectionIndex], theta->elements[i] - EPSILON);
        for (int prop = 1; prop < layers.size(); prop++) layers[prop]->activate(layers[prop - 1]);
        float cost1 = ((DenseLayer*)(layers[layers.size()-1]))->calcQuadraticCost(label->nodes);

        //recover value
        toModify->setH2D(i-sections[sectionIndex], theta->elements[i]);
        theta->elements[flattenSize + i] = (cost1 - cost0)/(2*EPSILON);
    }

    //calculate the sum of differences:
    double up = 0;           //grad - grad approx
    double down1 = 0;        //grad
    double down2 = 0;        //grad approx

    for(int i=0; i < flattenSize; i++){

        //difference = norm(grad - gradapprox) / norm(grad) + norm(gradapprox)
        up += pow(theta->elements[flattenSize*2 + i] - theta->elements[flattenSize*1 + i],2);
        down1 += pow(theta->elements[flattenSize*2 + i],2);
        down2 += pow(theta->elements[flattenSize*1 + i],2);
        //cout<<up<<" "<<down1<<" "<<down2<<endl;
    }

    Matrix::inspect(theta);
    divergence = sqrt(up)/(sqrt(down1) + sqrt(down2));
    return (float)divergence;
}
