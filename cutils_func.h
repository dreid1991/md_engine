#include "globalDefs.h"
#include "cutils_math.h"
#ifndef CUTILS_FUNC_H
#define CUTILS_FUNC_H

template <class T>
__device__ int baseNeighlistIdx(int *cumulSumMaxPerBlock, int warpSize) { 
    int cumulSumUpToMe = cumulSumMaxPerBlock[blockIdx.x];
    int maxNeighInMyBlock = cumulSumMaxPerBlock[blockIdx.x+1] - cumulSumUpToMe;
    int myWarp = threadIdx.x / warpSize;
    int myIdxInWarp = threadIdx.x % warpSize;
    return blockDim.x * cumulSumMaxPerBlock[blockIdx.x] + maxNeighInMyBlock * warpSize * myWarp + myIdxInWarp;
}

template <class T>
__device__ void copyToShared (T *src, T *dest, int n) {
    for (int i=threadIdx.x; i<n; i+=blockDim.x) {
        dest[i] = src[i];
    }
}
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
    int maxLookahead = log2f(blockDim.x-1);\
    for (int i=0; i<=maxLookahead; i++) {\
        int curLookahead = powf(2, i);\
        if (! (threadIdx.x % (curLookahead*2))) {\
            tmp[threadIdx.x] += tmp[threadIdx.x + curLookahead];\
        }\
        __syncthreads();\
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
    extern __shared__ K tmp[]; /*should have length of # threads in a block (PERBLOCK)*/\
    int potentialIdx = blockDim.x*blockIdx.x + threadIdx.x;\
    if (potentialIdx < n) {\
        unsigned int atomGroup = * (unsigned int *) &(fs[potentialIdx].w);\
        if (atomGroup & groupTag) {\
            tmp[threadIdx.x] = OPERATOR ( WRAPPER (src[blockDim.x*blockIdx.x + threadIdx.x]) ) ;\
            atomicAdd(dest+1, 1);\
        } else {\
            printf("no");\
            tmp[threadIdx.x] = 0;\
        }\
    } else {\
        tmp[threadIdx.x] = 0;\
    }\
    __syncthreads();\
    int maxLookahead = log2f(blockDim.x-1);\
    for (int i=0; i<=maxLookahead; i++) {\
        int curLookahead = powf(2, i);\
        if (! (threadIdx.x % (curLookahead*2))) {\
            tmp[threadIdx.x] += tmp[threadIdx.x + curLookahead];\
        }\
        __syncthreads();\
    }\
    if (threadIdx.x == 0) {\
        atomicAdd(dest, tmp[0]);\
    }\
}\

SUM_TAGS(sumVectorSqr3DTags, lengthSqr, make_float3);
#endif
