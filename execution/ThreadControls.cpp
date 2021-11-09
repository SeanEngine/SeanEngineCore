//
// Created by DanielSun on 11/8/2021.
//

#include <iostream>
#include "ThreadControls.h"

void ThreadControls::_syncThreads(dim3i threadId, dim3i blockSize, int *executionFlags) {
   int prev = executionFlags[threadId.y * blockSize.x + threadId.x];
   executionFlags[threadId.y * blockSize.x + threadId.x] = prev+1;
   begin:
   for (int i = 0; i < blockSize.x * blockSize.y; ++i) {
        if (executionFlags[i] == prev){
            goto begin;
        }
   }
}

void ThreadControls::_alloc(dim3i blockSize, void (*function)(vector<void*>*, dim3i, dim3i), vector<void *>* args) {

    for (int y = 0; y < blockSize.y; y++){
        for (int x = 0; x < blockSize.x; x++){
            dim3i threadId = dim3i(x, y);
            thread thr(function, args, blockSize, threadId);
            thr.detach();
        }
    }
}

void ThreadControls::_allocSynced(dim3i blockSize, void (*function)(vector<void*>*, dim3i, dim3i, int*), vector<void *>* args) {
   vector<thread> threads;
   int* executionFlags = (int*)malloc(sizeof(int)*blockSize.x * blockSize.y);
    for (int y = 0; y < blockSize.y; y++){
        for (int x = 0; x < blockSize.x; x++){
            dim3i threadId = dim3i(x, y);
            executionFlags[y * blockSize.x + x] = 0;
            threads.emplace_back(function, args, blockSize, threadId, executionFlags);
        }
    }
    for (auto & thread : threads){
        thread.join();
    }
    free(executionFlags);
}
