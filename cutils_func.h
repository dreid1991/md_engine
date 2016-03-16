#include "globalDefs.h"
#include "cutils_math.h"
#ifndef CUTILS_FUNC_H
#define CUTILS_FUNC_H
#define ATOMTEAMSIZE 4

//so this is for if you are execing one thread per atom, and you want its first entry in the neighborlist
inline __device__ int baseNeighlistIdxForAtom(int *cumulSumMaxPerBlock, int warpSize) {
    int atomIdx = GETIDX();
    int numAtomsPerBlockWhenTeams = blockDim.x / ATOMTEAMSIZE;
    int blockIdxWhenTeams = atomIdx / numAtomsPerBlockWhenTeams;
    int cumulSumUpToMe = cumulSumMaxPerBlock[blockIdxWhenTeams];
    int maxNeighInMyBlock = cumulSumMaxPerBlock[blockIdxWhenTeams+1] - cumulSumUpToMe;
    int threadIdxWhenTeams = atomIdx - blockIdxWhenTeams * numAtomsPerBlockWhenTeams;
    int numAtomsPerWarpWhenTeams = warpSize / ATOMTEAMSIZE;

    int myWarp = threadIdxWhenTeams / numAtomsPerWarpWhenTeams;
    int myIdxInWarp = threadIdxWhenTeams % numAtomsPerWarpWhenTeams;
    //okay, so cumul sums are rounded up to nearest multiple of ATOMTEAMSIZE (see gridgpu)
    return numAtomsPerBlockWhenTeams * cumulSumUpToMe + myWarp * maxNeighInMyBlock * numAtomsPerWarpWhenTeams + myIdxInWarp * ATOMTEAMSIZE;



}

//ONLY TO BE CALLED WITH ATOM TEAMS
inline __device__ int baseNeighlistIdx(int *cumulSumMaxPerBlock, int warpSize) {
    int cumulSumUpToMe = cumulSumMaxPerBlock[blockIdx.x]; //0
    int maxNeighInMyBlock = cumulSumMaxPerBlock[blockIdx.x+1] - cumulSumUpToMe; //24 or something
    int numAtomsPerWarpWhenTeams = warpSize / ATOMTEAMSIZE; //8
    int myWarp = threadIdx.x / warpSize; //0

    //int myIdxInWarp = (threadIdx.x % warpSize) / ATOMTEAMSIZE;
    //int myIdxInTeam = threadIdx.x % ATOMTEAMSIZE;

    return (blockDim.x/ATOMTEAMSIZE) * cumulSumMaxPerBlock[blockIdx.x] + myWarp * maxNeighInMyBlock * numAtomsPerWarpWhenTeams + threadIdx.x % warpSize;
}

//ONLY TO BE CALLED WITH ATOM TEAMS
inline __device__ int baseNeighlistIdxFromIndex(int *cumulSumMaxPerBlock, int warpSize, int atomIdx) {
    int numAtomsPerBlockWhenTeams = blockDim.x / ATOMTEAMSIZE; //64
    int blockIdxWhenTeams = atomIdx / numAtomsPerBlockWhenTeams; //0
    int cumulSumUpToMe = cumulSumMaxPerBlock[blockIdxWhenTeams];//00
    int maxNeighInMyBlock = cumulSumMaxPerBlock[blockIdxWhenTeams+1] - cumulSumUpToMe; //1?
    int threadIdxWhenTeams = atomIdx - blockIdxWhenTeams * numAtomsPerBlockWhenTeams; //0
    int numAtomsPerWarpWhenTeams = warpSize / ATOMTEAMSIZE; //8

    int myWarp = threadIdxWhenTeams / numAtomsPerWarpWhenTeams;//0

    int myIdxInWarp = threadIdxWhenTeams % numAtomsPerWarpWhenTeams;//0
    int myTeamIdx = threadIdx.x % ATOMTEAMSIZE;
    
    //printf("%d %d %d %d %d %d %d %d %d %d %d\n", threadIdx.x, atomIdx, numAtomsPerBlockWhenTeams, blockIdxWhenTeams, cumulSumUpToMe, maxNeighInMyBlock, threadIdxWhenTeams, numAtomsPerWarpWhenTeams, myWarp, myIdxInWarp, myTeamIdx);
    return numAtomsPerBlockWhenTeams * cumulSumUpToMe + myWarp * maxNeighInMyBlock * numAtomsPerWarpWhenTeams + myIdxInWarp * ATOMTEAMSIZE + myTeamIdx;
/*
    int blockIdx = idx / blockDim.x;
    int warpIdx = (idx - blockIdx * blockDim.x) / warpSize;
    int idxInWarp = idx - blockIdx * blockDim.x - warpIdx * warpSize;
    int cumSumUpToMyBlock = cumulSumMaxPerBlock[blockIdx];
    int perAtomMyWarp = cumulSumMaxPerBlock[blockIdx+1] - cumSumUpToMyBlock;
    int baseIdx = blockDim.x * cumSumUpToMyBlock + perAtomMyWarp * warpSize * warpIdx + idxInWarp;
    return baseIdx;
*/
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
//HEY - COULD OPTIMIZE BELOW BY STARTING curLookahead at 1 AND JUST MULTIPLYING BY 2 EACH TIME

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
        curLookahead *= 2;\
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
}\

SUM_TAGS(sumVectorSqr3DTags, lengthSqr, make_float3);
SUM_TAGS(sumVectorSqr3DTagsOverW, lengthSqrOverW, ); // for temperature



#endif
