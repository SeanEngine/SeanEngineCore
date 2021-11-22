//
// Created by DanielSun on 11/22/2021.
//

#ifndef CUDANNGEN2_DENSEMLP_H
#define CUDANNGEN2_DENSEMLP_H
#include "Model.cuh"

class DenseMLP : Model{
public:
    vector<Matrix>
    void loadDataSet(Engine::EngineConfig* cfg) override;
};


#endif //CUDANNGEN2_DENSEMLP_H
