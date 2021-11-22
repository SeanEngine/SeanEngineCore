#include <cstdio>
#include "utils/Matrix.cuh"
#include <windows.h>
#include <string>
#include <iostream>
#include "io/Reader.cuh"
#include "cublas_v2.h"
#include "execution/ThreadControls.h"

#pragma comment(lib, "cublas.lib")

using namespace std;


int main(int argc, char **argv) {
    vector<Matrix::Matrix2d *> mats;
    vector<Matrix::Matrix2d *> buf;
    unsigned char* buffer;
    unsigned char* bufCuda;
    string *fileNames;
    cudaMallocHost(&fileNames, sizeof(string) * 5000);
    cudaMallocHost(&buffer, sizeof(char)*12*2390);
    cudaMalloc(&bufCuda, sizeof(char)*12*2390);
    for (int i=0; i<5000; i++){
        Matrix::Matrix2d* matT;
        cudaMallocHost(&matT,sizeof(Matrix::Matrix2d));
        Matrix::callAllocElementH(matT, 28,28);
        mats.push_back(matT);
        fileNames[i] = string(R"(C:\Users\DanielSun\Desktop\resources\mnist\decompress_mnist\train\0\)") +
                to_string(i) + ".bmp";
    }
    for (int i=0;i<12;i++) {
        Matrix::Matrix2d* matT;
        cudaMallocHost(&matT,sizeof(Matrix::Matrix2d));
        Matrix::callAllocElementD(matT, 28,28);
        buf.push_back(matT);
    }
    int index = 0;

    LARGE_INTEGER cpuFreq;
    LARGE_INTEGER startTime;
    LARGE_INTEGER endTime;
    QueryPerformanceFrequency(&cpuFreq);
    QueryPerformanceCounter(&startTime);

    while(index < 5000){
        if (5000-index > 12) {
            Reader::readBMPFiles(12, fileNames, 2390, buffer, bufCuda, &mats, &buf, Reader::READ_GRAY, index);
        }else{
            Reader::readBMPFiles(5000-index, fileNames, 2390, buffer, bufCuda, &mats, &buf, Reader::READ_GRAY, index);
        }
        index+=12;
    }
    QueryPerformanceCounter(&endTime);
    long time = ((endTime.QuadPart - startTime.QuadPart));
    cout <<"5000 files :"<< time/10e6 << " s" << endl;
    Matrix::inspect(mats[25]);
}
