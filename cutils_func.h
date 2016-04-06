#pragma once
#ifndef CUTILS_FUNC_H
#define CUTILS_FUNC_H

#include "globalDefs.h"
#include "cutils_math.h"

template <class T>
__device__ int baseNeighlistIdx(int *cumulSumMaxPerBlock, int warpSize) { 
    int cumulSumUpToMe = cumulSumMaxPerBlock[blockIdx.x];
    int maxNeighInMyBlock = cumulSumMaxPerBlock[blockIdx.x+1] - cumulSumUpToMe;
    int myWarp = threadIdx.x / warpSize;
    int myIdxInWarp = threadIdx.x % warpSize;
    return blockDim.x * cumulSumMaxPerBlock[blockIdx.x] + maxNeighInMyBlock * warpSize * myWarp + myIdxInWarp;
}

template <class T>
__device__ int baseNeighlistIdxFromIndex(int *cumulSumMaxPerBlock, int warpSize, int idx) {
    int blockIdx = idx / blockDim.x;
    int warpIdx = (idx - blockIdx * blockDim.x) / warpSize;
    int idxInWarp = idx - blockIdx * blockDim.x - warpIdx * warpSize;
    int cumSumUpToMyBlock = cumulSumMaxPerBlock[blockIdx];
    int perAtomMyWarp = cumulSumMaxPerBlock[blockIdx+1] - cumSumUpToMyBlock;
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
inline __device__ void reduceByN(T *src, int span) { // where span is how many elements you're reducing over.  src had better be shared memory
    int maxLookahead = span / 2;
    int curLookahead = 1;
    while (curLookahead <= maxLookahead) {
        if (! (threadIdx.x % (curLookahead*2))) {
            src[threadIdx.x] += src[threadIdx.x + curLookahead];
        }
        curLookahead *= 2;
        __syncthreads();
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
template <class K, class T>\
__global__ void NAME (K *dest, T *src, int n) {\
    extern __shared__ K tmp[]; /*should have length of # threads in a block (PERBLOCK)*/\
    int potentialIdx = blockDim.x*blockIdx.x + threadIdx.x;\
    if (potentialIdx < n) {\
        tmp[threadIdx.x] = OPERATOR ( WRAPPER (src[blockDim.x*blockIdx.x + threadIdx.x]) ) ;\
    } else {\
       tmp[threadIdx.x] = 0;\
    }\
    __syncthreads();\
    int curLookahead = 1;\
    int maxLookahead = log2f(blockDim.x-1);\
    for (int i=0; i<=maxLookahead; i++) {\
        if (! (threadIdx.x % (curLookahead*2))) {\
            tmp[threadIdx.x] += tmp[threadIdx.x + curLookahead];\
        }\
        __syncthreads();\
        curLookahead *= 2;\
    }\
    if (threadIdx.x == 0) {\
        atomicAdd(dest, tmp[0]);\
    }\
}\


SUM(sumSingle, , );
SUM(sumVector, length, );
SUM(sumVectorSqr, lengthSqr, );
SUM(sumVectorSqr3D, lengthSqr, make_float3);

SUM(sumVector3D, length, make_float3);

//
#define SUM_TAGS(NAME, OPERATOR, WRAPPER) \
    template <class K, class T>\
__global__ void NAME (K *dest, T *src, int n, unsigned int groupTag, float4 *fs) {\
    extern __shared__ K tmp[]; /*should have length of # threads in a block (PERBLOCK)  */\
    int potentialIdx = blockDim.x*blockIdx.x + threadIdx.x;\
    if (potentialIdx < n) {\
        unsigned int atomGroup = * (unsigned int *) &(fs[potentialIdx].w);\
        if (atomGroup & groupTag) {\
            tmp[threadIdx.x] = OPERATOR ( WRAPPER (src[blockDim.x*blockIdx.x + threadIdx.x]) ) ;\
            atomicAdd(dest+1, 1);/*I TRIED DOING ATOMIC ADD IN SHARED MEMORY, BUT IT SET A BUNCH OF THE OTHER SHARED MEMORY VALUES TO ZERO.  VERY CONFUSING*/\
        } else {\
            tmp[threadIdx.x] = 0;\
        }\
    } else {\
        tmp[threadIdx.x] = 0;\
    }\
    __syncthreads();\
    int curLookahead = 1;\
    int maxLookahead = log2f(blockDim.x-1);\
    for (int i=0; i<=maxLookahead; i++) {\
        if (! (threadIdx.x % (curLookahead*2))) {\
            tmp[threadIdx.x] += tmp[threadIdx.x + curLookahead];\
        }\
        curLookahead *= 2;\
        __syncthreads();\
    }\
    if (threadIdx.x == 0) {\
        atomicAdd(dest, tmp[0]);\
    }\
}

SUM_TAGS(sumPlain, , );
SUM_TAGS(sumVectorSqr3DTags, lengthSqr, make_float3);
SUM_TAGS(sumVectorSqr3DTagsOverW, lengthSqrOverW, ); // for temperature

#endif
