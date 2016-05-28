#pragma once
#ifndef CUTILS_FUNC_H
#define CUTILS_FUNC_H

#include "globalDefs.h"
#include "cutils_math.h"
#define N_DATA_PER_THREAD 2 //must be power of 2, 4 found to be fastest for a floats and float4s
//Attenion please: tests show that N_DATA_PER_THREAD = 4 is faster, but it gives lower accuracy. Could reformulate in-thread adding to work like a parallel reduction, then accuracy should be the same

inline __device__ int baseNeighlistIdx(uint32_t *cumulSumMaxPerBlock, int warpSize) { 
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
__device__ void copyToShared (T *src, T *dest, int n) {
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
//Hey - if you pass warpsize, could avoid syncing al small lookaheads
#define SUM(NAME, OPERATOR, WRAPPER) \
template <class K, class T, int NPERTHREAD>\
__global__ void NAME (K *dest, T *src, int n, int warpSize) {\
    extern __shared__ K tmp[]; \
    const int copyBaseIdx = blockDim.x*blockIdx.x * NPERTHREAD + threadIdx.x;\
    const int copyIncrement = blockDim.x;\
    for (int i=0; i<NPERTHREAD; i++) {\
        int step = i * copyIncrement;\
        if (copyBaseIdx + step < n) {\
            tmp[threadIdx.x + step] = OPERATOR(WRAPPER((src[copyBaseIdx + step])));\
        } else {\
            tmp[threadIdx.x + step] = 0;\
        }\
    }\
    int curLookahead = NPERTHREAD;\
    int numLookaheadSteps = log2f(blockDim.x-1);\
    const int sumBaseIdx = threadIdx.x * NPERTHREAD;\
    __syncthreads();\
    for (int i=sumBaseIdx+1; i<sumBaseIdx + NPERTHREAD; i++) {\
        tmp[sumBaseIdx] += tmp[i];\
    }\
    for (int i=0; i<=numLookaheadSteps; i++) {\
        if (! (sumBaseIdx % (curLookahead*2))) {\
            tmp[sumBaseIdx] += tmp[sumBaseIdx + curLookahead];\
        }\
        if (curLookahead >= (NPERTHREAD * warpSize)) {\
            __syncthreads();\
        }\
        curLookahead *= 2;\
    }\
    if (threadIdx.x == 0) {\
        atomicAdd(dest, tmp[0]);\
    }\
}

SUM(sumSingle, , );
SUM(sumVector, length, );
SUM(sumVectorSqr, lengthSqr, );
SUM(sumVectorSqr3D, lengthSqr, make_float3);

SUM(sumVector3D, length, make_float3);
SUM(sumVectorSqr3DOverW, lengthSqrOverW, ); // for temperature


#define SUM_TAGS(NAME, OPERATOR, WRAPPER) \
template <class K, class T, int NPERTHREAD>\
__global__ void NAME (K *dest, T *src, int n, unsigned int groupTag, float4 *fs, int warpSize) {\
    extern __shared__ K tmp[]; \
    const int copyBaseIdx = blockDim.x*blockIdx.x * NPERTHREAD + threadIdx.x;\
    const int copyIncrement = blockDim.x;\
    int numAdded = 0;\
    for (int i=0; i<NPERTHREAD; i++) {\
        int step = i * copyIncrement;\
        if (copyBaseIdx + step < n) {\
            unsigned int atomGroup = * (unsigned int *) &(fs[copyBaseIdx + step].w);\
            if (atomGroup & groupTag) {\
                tmp[threadIdx.x + step] = OPERATOR ( WRAPPER( src[copyBaseIdx + step] ) );\
                numAdded++;\
            } else {\
                tmp[threadIdx.x + step] = 0;\
            }\
        } else {\
            tmp[threadIdx.x + step] = 0;\
        }\
    }\
    int curLookahead = NPERTHREAD;\
    int numLookaheadSteps = log2f(blockDim.x-1);\
    const int sumBaseIdx = threadIdx.x * NPERTHREAD;\
    atomicAdd(((int *) dest) + 1, numAdded);\
    __syncthreads();\
    for (int i=sumBaseIdx+1; i<sumBaseIdx + NPERTHREAD; i++) {\
        tmp[sumBaseIdx] += tmp[i];\
    }\
    for (int i=0; i<=numLookaheadSteps; i++) {\
        if (! (sumBaseIdx % (curLookahead*2))) {\
            tmp[sumBaseIdx] += tmp[sumBaseIdx + curLookahead];\
        }\
        if (curLookahead >= (NPERTHREAD * warpSize)) {\
            __syncthreads();\
        }\
        curLookahead *= 2;\
    }\
    if (threadIdx.x == 0) {\
        atomicAdd(dest, tmp[0]);\
    }\
}




SUM_TAGS(sumPlain, , );
SUM_TAGS(sumVectorSqr3DTags, lengthSqr, make_float3);
SUM_TAGS(sumVectorSqr3DTagsOverW, lengthSqrOverW, ); // for temperature
SUM_TAGS(sumVector3DTagsOverW, xyzOverW, ); //for linear momentum

#endif
