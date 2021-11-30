//
// Created by DanielSun on 11/8/2021.
//

#ifndef CUDANNGEN2_THREADCONTROLS_H
#define CUDANNGEN2_THREADCONTROLS_H

#include <utility>
#include <vector>
#include <thread>
#include <cstdarg>

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
 *     for other of your args, call them with arg indexes according to your parameters
 *     To code a TC function, using the following template:
 *
 *     for async :
 *     void func(vector<void*>* args, dim3i blockSize, dim3i threadId)
 *
 *     for sync :
 *     void func(vector<void*>* args, dim3i blockSize, dim3i threadId, int* executionFlags)
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

//this method detaches threads after creating them, I do not recommend using this though I provided the method
[[deprecated]] static void __alloc(dim3i blockSize, void (*function)(vector<void*>*, dim3i, dim3i), vector<void*>* args){
    ThreadControls::_alloc(blockSize, function, args);
}

static void __allocSynced(dim3i blockSize, void (*function)(vector<void*>*, dim3i, dim3i, int*), vector<void*>* args){
    ThreadControls::_allocSynced(blockSize, function, args);
}

static void __syncThreads(dim3i threadId, dim3i blockSize, int *executionFlags){
    ThreadControls::_syncThreads(threadId, blockSize, executionFlags);
}

// group the args
static vector<void*> __pack(int argc, void* ...){
    vector<void*> out;
    va_list c_arg{};
    va_start(c_arg, argc);
    out.reserve(argc);
    for (int i= 0; i < argc; i++) out.push_back(va_arg(c_arg, void*));
    va_end(c_arg);
    return out;
}


#endif //CUDANNGEN2_THREADCONTROLS_H
