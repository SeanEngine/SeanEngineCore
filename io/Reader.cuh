//
// Created by DanielSun on 11/17/2021.
//

#ifndef CUDANNGEN2_READER_CUH
#define CUDANNGEN2_READER_CUH
#include <string>
#include <vector>
#include "../execution/ThreadControls.h"
#include "../utils/Matrix.cuh"
#include <opencv2/opencv.hpp>
#include "opencv2/core/matx.hpp"

using namespace std;
class Reader {
public:
     static vector<string> getDirFiles(const string& path0);

     //read normal files in batch
     static unsigned char* readBytes(int fileCount, string* fileNames, int size, unsigned char* buffer);

     //read BMP files and post process them
     static void readImgGray(int threads, dim3i size, vector<string>* fileNames, vector<Matrix::Matrix2d *>* dataset);

     static void readImageRGB(int threads, dim3i size, vector<string>* fileNames, vector<Matrix::Tensor3d*>* dataset);
};


#endif //CUDANNGEN2_READER_CUH
