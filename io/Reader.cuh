//
// Created by DanielSun on 11/17/2021.
//

#ifndef CUDANNGEN2_READER_CUH
#define CUDANNGEN2_READER_CUH
#include <string>
#include <vector>
#include "../execution/ThreadControls.h"
#include "../utils/Matrix.cuh"

using namespace std;
class Reader {
public:
     enum Status{
         READ_RGB = 0,
         READ_GRAY = 1
     };
     static vector<string> getDirFiles(const string& path0);

     //read normal files in batch
     static unsigned char* readBytes(int fileCount, string* fileNames, int size, unsigned char* buffer);

     //read BMP files and post process them
     static void readBMPFiles(int threads, string* fileNames, int size, unsigned char* buffer, unsigned char *bufCuda,
                              vector<Matrix::Matrix2d*>* dataset, vector<Matrix::Matrix2d *>* outputBuf,
                              Status status, int offset, int offsetVec);
};


#endif //CUDANNGEN2_READER_CUH
