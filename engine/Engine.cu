//
// Created by DanielSun on 11/9/2021.
//

#include "Engine.cuh"



void Engine::boot(Engine::EngineConfig conf, Model* model) {
    this->config = conf;
    this->models = model;
}
