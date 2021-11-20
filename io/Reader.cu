//
// Created by DanielSun on 11/17/2021.
//

#include "Reader.cuh"
#include <cassert>

__global__ void genMat(Matrix::Matrix2d* target, unsigned char* buffer){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int index = (row*target->colcount) + col;
    unsigned char r = buffer[54 + index*4];
    unsigned char g = buffer[54 + index*4+1];
    unsigned char b = buffer[54 + index*4+2];
    target->set(row, col, static_cast<float>((int)r+(int)g+(int)b) / (256.0F * 3.0F));
}

void BMPProc(vector<void*>* args, dim3i blockSize, dim3i threadId, int* executionFlags){
    int readSize = *(int*)(*args)[0];
    auto* readBuffer = (unsigned char*)(*args)[1] + threadId.x * readSize;
    auto* bufCuda = (unsigned char*)(*args)[2] + threadId.x * readSize;
    auto* output = ( vector<Matrix::Matrix2d *>*)(*args)[3];
    auto* outputBuf = ( vector<Matrix::Matrix2d *>*)(*args)[4];

    Matrix::Matrix2d* dist = (*output)[threadId.x + *(int*)(*args)[5]];
    Matrix::Matrix2d* distBuf = (*outputBuf)[threadId.x + *(int*)(*args)[5]];

    assert(dist->rowcount * dist->colcount == (readSize-54)*4);  //match data size
    dim3 gridSize = dim3((dist->colcount + CUDA_BLOCK_SIZE.x - 1) / CUDA_BLOCK_SIZE.x,
                         (dist->rowcount + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y);
    cudaMemcpy(bufCuda, readBuffer, readSize, cudaMemcpyHostToDevice);
    genMat<<<gridSize, CUDA_BLOCK_SIZE>>>(distBuf, readBuffer);
    cudaDeviceSynchronize();
    cudaMemcpy(dist->elements, distBuf->elements, dist->colcount*dist->rowcount*sizeof(float), cudaMemcpyDeviceToHost);
}

void readFunc(vector<void*>* args, dim3i blockSize, dim3i threadId, int* executionFlags){
    int readSize = *(int*)(*args)[2];
    auto* readBuffer = (unsigned char*)(*args)[0] + threadId.x * readSize;
    auto* names = (string*)(*args)[1];
    FILE* file;
    fopen_s(&file,names[threadId.x].c_str(), "rb");
    fread(readBuffer,1,readSize,file);
    fclose(file);
}

unsigned char *Reader::readBytes(int fileCount, string* fileNames, int size, unsigned char* buffer) {
    vector<void*> params;
    params.push_back(buffer);
    params.push_back(fileNames);
    params.push_back(&size);
    __allocSynced(dim3i(fileCount,1),readFunc, &params);
    return buffer;
}

void Reader::readBMPFiles(int fileCount, string *fileNames, int size, unsigned char *buffer, unsigned char *bufCuda,
                                    vector<Matrix::Matrix2d *>* output, vector<Matrix::Matrix2d *>* outputBuf,
                                    Status status, int offset) {
    buffer = readBytes(fileCount, fileNames + offset, size, buffer);
    vector<void*> params;
    params.push_back(&size);
    params.push_back(buffer);
    params.push_back(bufCuda);
    params.push_back(output);
    params.push_back(outputBuf);
    params.push_back(&offset);
    switch (status) {
        case READ_RGB:break;
        case READ_GRAY:__allocSynced(dim3i(fileCount, 1), BMPProc, &params);break;
    }
}
