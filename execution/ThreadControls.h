//
// Created by DanielSun on 11/8/2021.
//

#ifndef CUDANNGEN2_THREADCONTROLS_H
#define CUDANNGEN2_THREADCONTROLS_H

#include <utility>
#include <vector>
#include <thread>

using namespace std;
struct dim3i{
    int x;
    int y;
    int z;

    dim3i(int x, int y){
        this->x = x;
        this->y = y;
    }

    dim3i(int x, int y, int z){
        this->x = x;
        this->y = y;
        this->z = z;
    }
};

/**
 * How this works:
 *     This system is designed to control cpu threads in the cuda fashion.
 *     In the arguments, the last 2 args are reserved for execution control, which are
 *     execution flags (int*), blockSize (dim3i) and threadIdx (dim3i). Thus, the actual argNum sending to
 *     each thread need to be 3 + your argNum, you should be aware of this while setting
 *     up your methods.
 *
 *     You should call the threadIdx by arg[argNum-1] and flags by arg[argNum-2]
 *
 *     for other of your args, call them with arg indexes according to your parameters
 *
 *     To code a TC function, using the following template:
 *
 *     for async :
 *     void func(vector<void*>* args, dim3i blockSize, dim3i threadId)
 */
class ThreadControls {
public:
    // start a matrix of threads just like with cuda
    static void _alloc(dim3i blockSize, void(*function)(vector<void*>*, dim3i, dim3i), vector<void*>* args);

    // choose whether you want the threads to stop the main thread before done (like cudaDeviceSynchronize())
    static void _allocSynced(dim3i blockSize, void (*function)(vector<void*>*, dim3i, dim3i, int*), vector<void*>* args);

    // sync all the threads
    static void _syncThreads(dim3i threadId, dim3i blockSize, int *executionFlags);
};

static void __alloc(dim3i blockSize, void (*function)(vector<void*>*, dim3i, dim3i), vector<void*>* args){
    ThreadControls::_alloc(blockSize, function, args);
}

static void __allocSynced(dim3i blockSize, void (*function)(vector<void*>*, dim3i, dim3i, int*), vector<void*>* args){
    ThreadControls::_allocSynced(blockSize, function, args);
}

static void __syncThreads(dim3i threadId, dim3i blockSize, int *executionFlags){
    ThreadControls::_syncThreads(threadId, blockSize, executionFlags);
}


#endif //CUDANNGEN2_THREADCONTROLS_H
