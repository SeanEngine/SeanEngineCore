//
// Created by DanielSun on 11/17/2021.
//

#include "Reader.cuh"
#include <cassert>
#include <iostream>
#include <io.h>
#include "../utils/logger.cuh"

__global__ void genMat(Matrix::Matrix2d* target, const unsigned char* buffer){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int index = (row*target->colcount) + col;
    unsigned char r = buffer[54 + index*3];
    unsigned char g = buffer[54 + index*3+1];
    unsigned char b = buffer[54 + index*3+2];
    target->set(row, col, static_cast<float>((int)r+(int)g+(int)b) / (256.0F * 3.0F));
}

void BMPProc(vector<void*>* args, dim3i blockSize, dim3i threadId, int* executionFlags){
    int readSize = *(int*)(*args)[0];
    auto* readBuffer = (unsigned char*)(*args)[1] + threadId.x * readSize;
    auto* bufCuda = (unsigned char*)(*args)[2] + threadId.x * readSize;
    auto* output =  ( vector<Matrix::Matrix2d *>*)(*args)[3];
    auto* outputBuf = ( vector<Matrix::Matrix2d *>*)(*args)[4];

    Matrix::Matrix2d* dist = (*output)[threadId.x + *(int*)(*args)[5]  + *(int*)(*args)[6]];
    Matrix::Matrix2d* distBuf = (*outputBuf)[threadId.x];

    //assert(dist->rowcount * dist->colcount*4 == (readSize-54));  //match data size
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

/**
 * The method for read the .bmp files
 * @param threads The amount of threads
 * @param fileNames The memory pointer that stores all the file names
 * @param size The size of each file that is going to be read
 * @param buffer The buffer on host memory (size equal to threads * size)
 * @param bufCuda The buffer on device memory (same size as buffer)
 * @param dataset The vec that stores all matrix (initialized)
 * @param outputBuf The vec that stores the matrix on device memory (size equal to threads)
 * @param status the method of reading (RGB or GRAY)
 * @param offset the offset of reading (equal to threads * the index of iteration)
 * @param offsetVec (optional) the total amount of files read in the previous iterations
 */
void Reader::readBMPFiles(int threads, string *fileNames, int size, unsigned char *buffer, unsigned char *bufCuda,
                          vector<Matrix::Matrix2d *>* dataset, vector<Matrix::Matrix2d *>* outputBuf,
                          Status status, int offset, int offsetVec) {
    buffer = readBytes(threads, fileNames + offset, size, buffer);

    vector<void*> params = __pack(7, &size, buffer, bufCuda, dataset, outputBuf, &offset, &offsetVec);
    switch (status) {
        case READ_RGB:break;
        case READ_GRAY:__allocSynced(dim3i(threads, 1), BMPProc, &params);break;
    }
}

vector<string> Reader::getDirFiles(const string& path0) {
    intptr_t hFile = 0;
    struct _finddata_t fileInfo{};
    string p;
    vector<string> temp;

    if ((hFile = _findfirst(p.assign(path0).append("\\*").c_str(), &fileInfo)) != -1){
        while (_findnext(hFile, &fileInfo) == 0){
            temp.push_back(p.assign(path0).append("\\").append(fileInfo.name));
        }
    }
    return temp;
}
