//
// Created by DanielSun on 11/17/2021.
//

#include "Reader.cuh"
#include <cassert>
#include <iostream>
#include <io.h>
#include "../utils/logger.cuh"
#include "../utils/Matrix.cuh"

__global__ void genMat(Matrix::Matrix2d* target, const uchar* in){
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    float value = row<target->rowcount && col<target->colcount ? (float)(in[row * target->colcount + col]) : 0.0f;
    target->set(row, col, value / 256.0f);
}

void ImgProcGray(vector<void*>* args, dim3i blockSize, dim3i threadId, int* executionFlags){
    auto* bytesBuf = (uchar*)(*args)[3];
    auto* dataset = (vector<Matrix::Matrix2d*>*)(*args)[2];
    vector<string> filenames = *(vector<string>*)(*args)[1];
    dim3i size = *(dim3i*)(*args)[0];
    int part = (int)filenames.size()/blockSize.x;
    dim3 gridSize = dim3((size.x + CUDA_BLOCK_SIZE.x - 1) / (CUDA_BLOCK_SIZE.x),
                         (size.y + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    int dataIndex = (int)(dataset->size() - filenames.size());

    for (int i = threadId.x * part; i < (threadId.x == blockSize.x-1 ? filenames.size()-1 : (threadId.x+1)*part); i++){
        string target = filenames[i];
        cv::Mat img = cv::imread(target, cv::IMREAD_GRAYSCALE);
        assert(!img.empty());

        uchar* proc = bytesBuf + size.x * size.y * threadId.x;
        cudaMemcpy(proc, img.data, sizeof(uchar) * size.x * size.y, cudaMemcpyHostToDevice);

        Matrix::Matrix2d* mat1 = (*dataset)[dataIndex + i];
        genMat<<<gridSize, CUDA_BLOCK_SIZE>>>(mat1,proc);
        cudaDeviceSynchronize();
    }
}

void readFunc(vector<void*>* args, dim3i blockSize, dim3i threadId, int* executionFlags){
    int readSize = *(int*)(*args)[2];
    auto* readBuffer = (unsigned char*)(*args)[0] + threadId.x * readSize;
    auto* names = (string*)(*args)[1];
    FILE* file;
    fopen_s(&file,names[threadId.x].c_str(), "rb+");
    if(file== nullptr)return;
    fread_s(readBuffer,readSize,1,readSize,file);
    fclose(file);
}

unsigned char *Reader::readBytes(int fileCount, string* fileNames, int size, unsigned char* buffer) {
    vector<void*> params = __pack(3,buffer,fileNames,&size);
    __allocSynced(dim3i(fileCount,1),readFunc, &params);
    return buffer;
}

void Reader::readImgGray(int threads, dim3i size, vector<string>* fileNames, vector<Matrix::Matrix2d *>* dataset) {
    uchar* bytesBuf;
    cudaMalloc(&bytesBuf, sizeof(uchar) * size.x * size.y * threads);
    vector<void*> params = __pack(4, &size, fileNames, dataset, bytesBuf);
    __allocSynced(dim3i(threads, 1), ImgProcGray, &params);
    cudaFree(bytesBuf);
}

vector<string> Reader::getDirFiles(const string& path0) {
    intptr_t hFile = 0;
    struct _finddata_t fileInfo{};
    string p;
    vector<string> temp;

    if ((hFile = _findfirst(p.assign(path0).append("\\*").c_str(), &fileInfo)) != -1){
        while (_findnext(hFile, &fileInfo) == 0){
            if(string(fileInfo.name) == "." || string(fileInfo.name) == "..") continue;
            temp.push_back(p.assign(path0).append("\\").append(fileInfo.name));
        }
    }
    return temp;
}
