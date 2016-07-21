#pragma once
#ifndef CUTILS_FUNC_H
#define CUTILS_FUNC_H

#include "globalDefs.h"
#include "cutils_math.h"
#include "Virial.h"
#include "SharedMem.h"
#define N_DATA_PER_THREAD 4 //must be power of 2, 4 found to be fastest for a floats and float4s
//tests show that N_DATA_PER_THREAD = 4 is fastest

inline __device__ int baseNeighlistIdx(const uint32_t *cumulSumMaxPerBlock, int warpSize) { 
    uint32_t cumulSumUpToMe = cumulSumMaxPerBlock[blockIdx.x];
    uint32_t maxNeighInMyBlock = cumulSumMaxPerBlock[blockIdx.x+1] - cumulSumUpToMe;
    int myWarp = threadIdx.x / warpSize;
    int myIdxInWarp = threadIdx.x % warpSize;
    return blockDim.x * cumulSumMaxPerBlock[blockIdx.x] + maxNeighInMyBlock * warpSize * myWarp + myIdxInWarp;
}

inline __device__ int baseNeighlistIdxFromIndex(uint32_t *cumulSumMaxPerBlock, int warpSize, int idx) {
    int blockIdx = idx / blockDim.x;
    int warpIdx = (idx - blockIdx * blockDim.x) / warpSize;
    int idxInWarp = idx - blockIdx * blockDim.x - warpIdx * warpSize;
    uint32_t cumSumUpToMyBlock = cumulSumMaxPerBlock[blockIdx];
    uint32_t perAtomMyWarp = cumulSumMaxPerBlock[blockIdx+1] - cumSumUpToMyBlock;
    int baseIdx = blockDim.x * cumSumUpToMyBlock + perAtomMyWarp * warpSize * warpIdx + idxInWarp;
    return baseIdx;

}

template <class T>
__device__ void copyToShared (const T *src, T *dest, int n) {
    for (int i=threadIdx.x; i<n; i+=blockDim.x) {
        dest[i] = src[i];
    }
}
template <class T>
inline __device__ void reduceByN(T *src, int span, int warpSize) { // where span is how many elements you're reducing over.  src had better be shared memory
    int maxLookahead = span / 2;
    int curLookahead = 1;
    while (curLookahead <= maxLookahead) {
        if (! (threadIdx.x % (curLookahead*2))) {
            src[threadIdx.x] += src[threadIdx.x + curLookahead];
        }
        curLookahead *= 2;
        if (curLookahead >= warpSize) {
            __syncthreads();
        }
    }
}
template <class T>
inline __device__ void maxByN(T *src, int span, int warpSize) { // where span is how many elements you're reducing over.  src had better be shared memory
    int maxLookahead = span / 2;
    int curLookahead = 1;
    while (curLookahead <= maxLookahead) {
        if (! (threadIdx.x % (curLookahead*2))) {
            T a = src[threadIdx.x];
            T b = src[threadIdx.x + curLookahead];
            //max isn't defined for all types in cuda
            if (a > b) {
                src[threadIdx.x] = a;
            } else {
                src[threadIdx.x] = b;
            }
        }
        curLookahead *= 2;
        if (curLookahead >= warpSize) {
            __syncthreads();
        }
    }
}
//only safe to use if reducing within a warp
template <class T>
inline __device__ void reduceByN_NOSYNC(T *src, int span) { // where span is how many elements you're reducing over.  src had better be shared memory
    int maxLookahead = span / 2;
    int curLookahead = 1;
    while (curLookahead <= maxLookahead) {
        if (! (threadIdx.x % (curLookahead*2))) {
            src[threadIdx.x] += src[threadIdx.x + curLookahead];
        }
        curLookahead *= 2;
    }
}

#define ACCUMULATION_CLASS(NAME, TO, FROM, VARNAME_PROC, PROC, ZERO)\
class NAME {\
public:\
    inline __host__ __device__ TO process (FROM & VARNAME_PROC ) {\
        return ( PROC );\
    }\
    inline __host__ __device__ TO zero() {\
        return ( ZERO );\
    }\
};
  //          vir.vals[3] = vel.x * vel.y;
   //         vir.vals[4] = vel.x * vel.z;
    //        vir.vals[5] = vel.y * vel.z;

ACCUMULATION_CLASS(SumSingle, float, float, x, x, 0);
ACCUMULATION_CLASS(SumSqr, float, float, x, x*x, 0);
ACCUMULATION_CLASS(SumVectorSqr3D, float, float4, v, lengthSqr(make_float3(v)), 0);
ACCUMULATION_CLASS(SumVectorSqr3DOverW, float, float4, v, lengthSqrOverW(v), 0); //for temperature
ACCUMULATION_CLASS(SumVectorXYZOverW, float4, float4, v, xyzOverW(v), make_float4(0, 0, 0, 0)); //for linear momentum
ACCUMULATION_CLASS(SumVectorToVirial, Virial, float4, v, Virial(v.x*v.x, v.y*v.y, v.z*v.z, v.x*v.y, v.x*v.z, v.y*v.z), Virial(0, 0, 0, 0, 0, 0)); 

template <class K, class T, class C, int NPERTHREAD>
__global__ void accumulate_gpu(K *dest, T *src, int n, int warpSize, C instance) {
    SharedMemory<K> sharedMem;
    K *tmp = sharedMem.getPointer();
    
    const int copyBaseIdx = blockDim.x*blockIdx.x * NPERTHREAD + threadIdx.x;
    const int copyIncrement = blockDim.x;
    for (int i=0; i<NPERTHREAD; i++) {
        int step = i * copyIncrement;
        if (copyBaseIdx + step < n) {
            tmp[threadIdx.x + step] = instance.process(src[copyBaseIdx + step]);
        } else {
            tmp[threadIdx.x + step] = instance.zero();
        }
    }
    int curLookahead = NPERTHREAD;
    int numLookaheadSteps = log2f(blockDim.x-1);
    const int sumBaseIdx = threadIdx.x * NPERTHREAD;
    __syncthreads();
    for (int i=sumBaseIdx+1; i<sumBaseIdx + NPERTHREAD; i++) {
        tmp[sumBaseIdx] += tmp[i];
    }
    for (int i=0; i<=numLookaheadSteps; i++) {
        if (! (sumBaseIdx % (curLookahead*2))) {
            tmp[sumBaseIdx] += tmp[sumBaseIdx + curLookahead];
        }
        curLookahead *= 2;
        if (curLookahead >= (NPERTHREAD * warpSize)) {
            __syncthreads();
        }
    }
    if (threadIdx.x < sizeof(K) / sizeof(float)) {
        //one day, some hero will find out why it doesn't work to do atomicAdd as a member of the accumulation class.
        //in the mean time, just adding 32 bit chunks.  Could template this to do ints too.
        float *destFloat = (float *) dest;
        float *tmpFloat = (float *) tmp;
        atomicAdd(destFloat + threadIdx.x, tmpFloat[threadIdx.x]);
    }
}
//dealing with the common case of summing based on group tags
#define ACCUMULATION_CLASS_IF(NAME, TO, FROM, VARNAME_PROC, PROC, ZERO)\
class NAME {\
public:\
    float4 *fs;\
    uint32_t groupTag;\
    NAME(float4 *fs_, uint32_t groupTag_) : fs(fs_), groupTag(groupTag_) {}\
    inline __host__ __device__ TO process (FROM & VARNAME_PROC ) {\
        return ( PROC );\
    }\
    inline __host__ __device__ TO zero() {\
        return ( ZERO );\
    }\
    inline __host__ __device__ bool willProcess(FROM *src, int idx) {\
        uint32_t atomGroupTag = * (uint32_t *) &(fs[idx].w);\
        return atomGroupTag & groupTag;\
    }\
};
ACCUMULATION_CLASS_IF(SumSingleIf, float, float, x, x, 0);
ACCUMULATION_CLASS_IF(SumSqrIf, float, float, x, x*x, 0);
ACCUMULATION_CLASS_IF(SumVectorSqr3DIf, float, float4, v, lengthSqr(make_float3(v)), 0);
ACCUMULATION_CLASS_IF(SumVectorSqr3DOverWIf, float, float4, v, lengthSqrOverW(v), 0); //for temperature
ACCUMULATION_CLASS_IF(SumVectorXYZOverWIf, float4, float4, v, xyzOverW(v), make_float4(0, 0, 0, 0)); //for linear momentum
ACCUMULATION_CLASS_IF(SumVectorToVirialIf, Virial, float4, v, Virial(v.x*v.x, v.y*v.y, v.z*v.z, v.x*v.y, v.x*v.z, v.y*v.z), Virial(0, 0, 0, 0, 0, 0)); 




template <class K, class T, class C, int NPERTHREAD>
__global__ void accumulate_gpu_if(K *dest, T *src, int n, int warpSize, C instance) {
    SharedMemory<K> sharedMem;
    K *tmp = sharedMem.getPointer();

    int numAdded = 0;
    const int copyBaseIdx = blockDim.x*blockIdx.x * NPERTHREAD + threadIdx.x;
    const int copyIncrement = blockDim.x;
    for (int i=0; i<NPERTHREAD; i++) {
        int step = i * copyIncrement;
        if (copyBaseIdx + step < n) {
            int curIdx = copyBaseIdx + step;
            if (instance.willProcess(src, curIdx)) { //can deal with bounds here
                numAdded ++;
                tmp[threadIdx.x + step] = instance.process(src[curIdx]);
            } else {
                tmp[threadIdx.x + step] = instance.zero();
            }
        } else {
            tmp[threadIdx.x + step] = instance.zero();
        }
    }
    int curLookahead = NPERTHREAD;
    int numLookaheadSteps = log2f(blockDim.x-1);
    const int sumBaseIdx = threadIdx.x * NPERTHREAD;
    atomicAdd((int *) (dest + 1), numAdded);
    __syncthreads();
    for (int i=sumBaseIdx+1; i<sumBaseIdx + NPERTHREAD; i++) {
        tmp[sumBaseIdx] += tmp[i];
    }
    for (int i=0; i<=numLookaheadSteps; i++) {
        if (! (sumBaseIdx % (curLookahead*2))) {
            tmp[sumBaseIdx] += tmp[sumBaseIdx + curLookahead];
        }
        curLookahead *= 2;
        if (curLookahead >= (NPERTHREAD * warpSize)) {
            __syncthreads();
        }
    }
    if (threadIdx.x < sizeof(K) / sizeof(float)) {
        //one day, some hero will find out why it doesn't work to do atomicAdd as a member of the accumulation class.
        //in the mean time, just adding 32 bit chunks.  Could template this to do ints too.
        float *destFloat = (float *) dest;
        float *tmpFloat = (float *) tmp;
        atomicAdd(destFloat + threadIdx.x, tmpFloat[threadIdx.x]);
    }
}

#endif
