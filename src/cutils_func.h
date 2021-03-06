#pragma once
#ifndef CUTILS_FUNC_H
#define CUTILS_FUNC_H

#include "globalDefs.h"
#include "cutils_math.h"
#include "Virial.h"
#include "SharedMem.h"
#define N_DATA_PER_THREAD 4 //must be power of 2, 4 found to be fastest for a reals and real4s
//tests show that N_DATA_PER_THREAD = 4 is fastest

#ifdef __CUDACC__
inline __device__ int baseNeighlistIdx(const uint32_t *cumulSumMaxMemPerWarp, int warpSize, int nThreadPerAtom) { 
    uint32_t cumulSumUpToMe = cumulSumMaxMemPerWarp[blockIdx.x];
    uint32_t memSizePerWarpMe = cumulSumMaxMemPerWarp[blockIdx.x+1] - cumulSumUpToMe;
    int warpsPerBlock = blockDim.x/warpSize;
    int myWarp = threadIdx.x / warpSize;
    int myIdxInWarp = threadIdx.x % warpSize;
    return warpsPerBlock * cumulSumUpToMe + memSizePerWarpMe * myWarp + myIdxInWarp;
}

//int baseIdx_globalJIdx = baseNeighlistIdxFromRPIndex(cumulSumMaxPerBlock, warpSize, globalJIdx, warpSize);
inline __device__ int baseNeighlistIdxFromRPIndex(const uint32_t *cumulSumMaxMemPerWarp, int warpSize, int myRingPolyIdx, int nThreadPerAtom) { 
    int nAtomPerBlock = blockDim.x / nThreadPerAtom;
    int      blockIdx           = myRingPolyIdx / nAtomPerBlock;
    uint32_t cumulSumUpToMe     = cumulSumMaxMemPerWarp[blockIdx];
    uint32_t memSizePerWarpMe   = cumulSumMaxMemPerWarp[blockIdx+1] - cumulSumUpToMe;
    int nthAtomInBlock          = myRingPolyIdx % nAtomPerBlock;
    int nAtomPerWarp            = warpSize / nThreadPerAtom;
    int myWarp                  = nthAtomInBlock / nAtomPerWarp;
    int myIdxInWarp             = nthAtomInBlock % nAtomPerWarp;
    int warpsPerBlock           = blockDim.x/warpSize;
    return warpsPerBlock * cumulSumUpToMe + memSizePerWarpMe * myWarp + myIdxInWarp * nThreadPerAtom;
}

inline __device__ int baseNeighlistIdxFromRPIndex(const uint32_t *cumulSumMaxMemPerWarp, int warpSize, int myRingPolyIdx) {
    int      blockIdx           = myRingPolyIdx / blockDim.x;
    uint32_t cumulSumUpToMe     = cumulSumMaxMemPerWarp[blockIdx];
    uint32_t memSizePerWarpMe   = cumulSumMaxMemPerWarp[blockIdx+1] - cumulSumUpToMe;
    int nthAtomInBlock          = myRingPolyIdx % blockDim.x;
    int myWarp                  = nthAtomInBlock / warpSize;
    int myIdxInWarp             = nthAtomInBlock % warpSize;
    int warpsPerBlock           = blockDim.x/warpSize;
    return warpsPerBlock * cumulSumUpToMe + memSizePerWarpMe * myWarp + myIdxInWarp;
}
/*
inline __device__ int baseNeighlistIdxFromRPIndex(const uint32_t *cumulSumMaxMemPerWarp, int warpSize, int myRingPolyIdx, int nThreadPerAtom) { 
    int      blockIdx           = myRingPolyIdx / blockDim.x;
    uint32_t cumulSumUpToMe     = cumulSumMaxMemPerWarp[blockIdx];
    uint32_t memSizePerWarpMe = cumulSumMaxMemPerWarp[blockIdx+1] - cumulSumUpToMe;
    int      myWarp             = ( myRingPolyIdx % blockDim.x) / warpSize;
    int      myIdxInWarp        = myRingPolyIdx % warpSize;
    int warpsPerBlock = blockDim.x/warpSize;
    return warpsPerBlock * cumulSumUpToMe + memSizePerWarpMe * myWarp + myIdxInWarp;
}
*/
inline __device__ int baseNeighlistIdxFromIndex(const uint32_t *cumulSumMaxPerBlock, int warpSize, int idx) {
    int blockIdx = idx / blockDim.x; // idx as from GETIDX(); but, as long as idx is an absolute metric, 
    // we are ok
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



//toSort should be of size blockDim.x, sorted should be of size nWarp, sortSize <= warpSize
template
<class T>
inline __device__ void sortBubble_NOSYNC(T *toSort, bool *sorted, int sortSize, int warpSize) {
    int warpIdx = threadIdx.x / warpSize;
    sorted[warpIdx] = false; //so each warp does its own thing  
    bool amLast = (threadIdx.x + 1)%sortSize == 0; //I am the last thread in my sorting chunk, I don't need to do anything
    bool amEven = threadIdx.x%2 == 0;
    while (sorted[warpIdx] == false) {
        sorted[warpIdx] = true;
        bool didSwap = false;
        if (!amEven and !amLast) {
            if (toSort[threadIdx.x+1] < toSort[threadIdx.x]) {
                T tmp = toSort[threadIdx.x+1];
                toSort[threadIdx.x+1] = toSort[threadIdx.x];
                toSort[threadIdx.x] = tmp;
                didSwap = true;
            }
        }
        if (amEven) {
            if (toSort[threadIdx.x+1] < toSort[threadIdx.x]) {
                T tmp = toSort[threadIdx.x+1];
                toSort[threadIdx.x+1] = toSort[threadIdx.x];
                toSort[threadIdx.x] = tmp;
                didSwap = true;
            }
        }
        if (didSwap) {
            sorted[warpIdx] = false;
        }
    }
}


#ifdef DASH_DOUBLE
#define ACCUMULATION_CLASS(NAME, TO, FROM, VARNAME_PROC, PROC, ZERO)\
class NAME {\
public:\
    inline __host__ __device__ TO process (FROM & VARNAME_PROC ) {\
        return ( PROC );\
    }\
    inline __host__ __device__ TO zero() {\
        return ( ZERO );\
    }\
};\
\
class NAME ## If {\
public:\
    double4  *fs;\
    uint32_t groupTag;\
    NAME ## If (double4 *fs_, uint32_t groupTag_) : fs(fs_), groupTag(groupTag_) {}\
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


#else /* DASH_DOUBLE */

#define ACCUMULATION_CLASS(NAME, TO, FROM, VARNAME_PROC, PROC, ZERO)\
class NAME {\
public:\
    inline __host__ __device__ TO process (FROM & VARNAME_PROC ) {\
        return ( PROC );\
    }\
    inline __host__ __device__ TO zero() {\
        return ( ZERO );\
    }\
};\
\
class NAME ## If {\
public:\
    float4 *fs;\
    uint32_t groupTag;\
    NAME ## If (float4 *fs_, uint32_t groupTag_) : fs(fs_), groupTag(groupTag_) {}\
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

#endif /* DASH_DOUBLE */

ACCUMULATION_CLASS(SumSingle, real, real, x, x, 0);
ACCUMULATION_CLASS(SumVirial, Virial, Virial, vir, vir, Virial(0, 0, 0, 0, 0, 0));
ACCUMULATION_CLASS(SumSqr, real, real, x, x*x, 0);
ACCUMULATION_CLASS(SumVectorSqr3D, real, real4, v, lengthSqr(make_real3(v)), 0);
ACCUMULATION_CLASS(SumVectorSqr3DOverW, real, real4, v, lengthSqrOverW(v), 0); //for temperature
ACCUMULATION_CLASS(SumVectorXYZOverW, real4, real4, v, xyzOverW(v), make_real4(0, 0, 0, 0)); //for linear momentum
//opt by precomputing 1/w.  probably trivial speedup
ACCUMULATION_CLASS(SumVectorToVirial, Virial, real4, v, Virial(v.x*v.x, v.y*v.y, v.z*v.z, v.x*v.y, v.x*v.z, v.y*v.z), Virial(0, 0, 0, 0, 0, 0)); 

/* TODO: this is the line giving grief for massless particles! */
ACCUMULATION_CLASS(SumVectorToVirialOverW, Virial, real4, v, Virial(v.x*v.x/v.w, v.y*v.y/v.w, v.z*v.z/v.w, v.x*v.y/v.w, v.x*v.z/v.w, v.y*v.z/v.w), Virial(0, 0, 0, 0, 0, 0)); 
ACCUMULATION_CLASS(SumVirialToScalar, real, Virial, vir, (vir[0]+vir[1]+vir[2]), 0); 

template <class K, class T, class C, int NPERTHREAD>
__global__ void oneToOne_gpu(K *dest, T *src, int n, C instance) {
    
    const int copyBaseIdx = blockDim.x*blockIdx.x * NPERTHREAD + threadIdx.x;
    const int copyIncrement = blockDim.x;
    for (int i=0; i<NPERTHREAD; i++) {
        int step = i * copyIncrement;
        if (copyBaseIdx + step < n) {
            dest[copyBaseIdx + step] = instance.process(src[copyBaseIdx + step]);
            
        } 
    }
}


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
    if (threadIdx.x < sizeof(K) / sizeof(real)) {
        //one day, some hero will find out why it doesn't work to do atomicAdd as a member of the accumulation class.
        //in the mean time, just adding 32 bit chunks.  Could template this to do ints too.
#ifdef DASH_DOUBLE
        double *destDASH = (double *) dest;
        double *tmpDASH = (double *) tmp;
#else
        float *destDASH = (float *) dest;
        float *tmpDASH = (float *) tmp;
#endif /* DASH_DOUBLE */


        /* NOTE TO FUTURE PEOPLE: 
         * We use the macro above to branch between double and single, because trying to cast as 'real'
         * as shown below causes the compiler to throw an error;
         * types deduced in the atomicAdd operations are (real *, real) instead of 
         * whatever real was compiled to be via typedef in globalDefs.h - either float, or double.
         * This is an issue because, strictly speaking, there is no atomicAdd function that takes 
         * parameters of type (real *, real) - but the compiler doesn't recognize them as double or float.
         */
        //real *destreal = (real *) dest;
        //real *tmpreal = (real *) tmp;
        
        
        atomicAdd(destDASH + threadIdx.x, tmpDASH[threadIdx.x]);
        //atomicAdd(destreal + threadIdx.x, tmpreal[threadIdx.x]);

    }
}

//dealing with the common case of summing based on group tags
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
    if (threadIdx.x < sizeof(K) / sizeof(real)) {
        //one day, some hero will find out why it doesn't work to do atomicAdd as a member of the accumulation class.
        //in the mean time, just adding 32 bit chunks.  Could template this to do ints too.
#ifdef DASH_DOUBLE
        double *destDASH = (double *) dest;
        double *tmpDASH = (double *) tmp;
#else
        float *destDASH = (float *) dest;
        float *tmpDASH = (float *) tmp;
#endif /* DASH_DOUBLE */
        atomicAdd(destDASH + threadIdx.x, tmpDASH[threadIdx.x]);
    }
}


#if __CUDACC_VER_MAJOR__ >= 9
// verified correct
__inline__ __device__ double3 warpReduceSum(double3 val, int warpSize) {
    // 0xffffffff is bitmask saying all threads participate in warp reduction
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val.x += __shfl_down_sync(0xffffffff,val.x, offset);
        val.y += __shfl_down_sync(0xffffffff,val.y, offset);
        val.z += __shfl_down_sync(0xffffffff,val.z, offset);
    }
    return val;
}

__inline__ __device__ float3 warpReduceSum(float3 val, int warpSize) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val.x += __shfl_down_sync(0xffffffff,val.x, offset);
        val.y += __shfl_down_sync(0xffffffff,val.y, offset);
        val.z += __shfl_down_sync(0xffffffff,val.z, offset);
    }
    return val;
}

__inline__ __device__ Virial warpReduceSum(Virial val, int warpSize) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val[0] += __shfl_down_sync(0xffffffff,val[0],offset);
        val[1] += __shfl_down_sync(0xffffffff,val[1],offset);
        val[2] += __shfl_down_sync(0xffffffff,val[2],offset);
        val[3] += __shfl_down_sync(0xffffffff,val[3],offset);
        val[4] += __shfl_down_sync(0xffffffff,val[4],offset);
        val[5] += __shfl_down_sync(0xffffffff,val[5],offset);
    }
    return val;
}
#endif /* #if __CUDACC_VER_MAJOR__ >= 9 */

#endif /* #ifdef __CUDACC__ */

#endif
