#include "GridGPU.h"
#include <set>
#include "State.h"
#include "helpers.h"
#include "Bond.h"
#include "BoundsGPU.h"
#include "list_macro.h"
#include "Mod.h"
#include "Fix.h"
#include "cutils_func.h"
//for debugging
__global__ void countNumInGridCells(float4 *xs, int nAtoms, int *counts, int *atomIdxs, float3 os, float3 ds, int3 ns) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        //printf("idx %d\n", idx);
        int3 sqrIdx = make_int3((make_float3(xs[idx]) - os) / ds);
        int sqrLinIdx = LINEARIDX(sqrIdx, ns);
        //printf("lin is %d\n", sqrLinIdx);
        int myPlaceInGrid;
        myPlaceInGrid = atomicAdd(counts + sqrLinIdx, 1); //atomicAdd returns old value
        //printf("grid is %d\n", myPlaceInGrid);
        //printf("myPlaceInGrid %d\n", myPlaceInGrid);
        atomIdxs[idx] = myPlaceInGrid;
        //okay - atoms seem to be getting assigned the right idx in grid 
    }
}



__global__ void periodicWrap(float4 *xs, int nAtoms, BoundsGPU bounds) {
    int idx = GETIDX();
    if (idx < nAtoms) {

        float4 pos = xs[idx];

        float id = pos.w;
        float3 trace = bounds.trace();
        float3 diffFromLo = make_float3(pos) - bounds.lo;
        float3 imgs = floorf(diffFromLo / trace); //are unskewed at this point
        float3 pos_orig = make_float3(pos);
        pos -= make_float4(trace * imgs * bounds.periodic);
        pos.w = id;
        //if (not(pos.x==orig.x and pos.y==orig.y and pos.z==orig.z)) { //sigh
        if (imgs.x != 0 or imgs.y != 0 or imgs.z != 0) {
            xs[idx] = pos;
        }

    }

}
#define TESTIDX 1000
/*
__global__ void printFloats(cudaTextureObject_t xs, int n) {
    int idx = GETIDX();
    if (idx < n) {
        int xIdx = XIDX(idx);
        int yIdx = YIDX(idx);
        float4 x = tex2D<float4>(xs, xIdx, yIdx);
        printf("idx %d, vals %f %f %f\n", idx, x.x, x.y, x.z);

    }
}

__global__ void printInts(cudaTextureObject_t xs, int n) {
    int idx = GETIDX();
    if (idx < n) {
        int xIdx = XIDX(idx);
        int yIdx = YIDX(idx);
        int x = tex2D<int>(xs, xIdx, yIdx);
        printf("idx %d, val %d\n", idx, x);

    }
}


__global__ void printIntsArray(int *xs, int n) {
    int idx = GETIDX();
    if (idx < n) {
        int x = xs[idx];
        printf("idx %d, val %d\n", idx, x);
    }
}


__global__ void printNeighbors(int *neighborlistBounds, cudaTextureObject_t neighbors, int nAtoms) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        int begin = neighborlistBounds[idx];
        int end = neighborlistBounds[idx+1];
        for (int i=begin; i<end; i++) {
            int xIdx = XIDX(i);
            int yIdx = YIDX(i);
            int x = tex2D<int>(neighbors, xIdx, yIdx);
            printf("idx %d has neighbor of idx %d\n", idx, x);
        }
    }
}
*/
template <typename T>
__device__ void copyToOtherSurf(cudaSurfaceObject_t from, cudaSurfaceObject_t to, int idx_init, int idx_final) {
    int xIdx, yIdx, xAddr;
    xIdx = XIDX(idx_init, sizeof(T));
    yIdx = YIDX(idx_init, sizeof(T));
    xAddr = xIdx * sizeof(T);
    T val = surf2Dread<T>(from, xAddr, yIdx);
    xIdx = XIDX(idx_final, sizeof(T));
    yIdx = YIDX(idx_final, sizeof(T));
    xAddr = xIdx * sizeof(T);
    surf2Dwrite(val, to, xAddr, yIdx);
}

template <typename T>
__device__ void copyToOtherList(T *from, T *to, int idx_init, int idx_final) {
    to[idx_final] = from[idx_init];
}

__global__ void sortPerAtomArrays(
        float4 *xsFrom,         float4 *xsTo, 
        float4  *vsFrom,        float4 *vsTo,
        float4  *fsFrom,        float4 *fsTo,
        float4  *fsLastFrom,    float4 *fsLastTo,
        uint *idsFrom, uint *idsTo,
        float *qsFrom, float *qsTo,

        cudaSurfaceObject_t idToIdx,
        int *gridCellArrayIdxs, int *idxInGridCell, int nAtoms, float3 os, float3 ds, int3 ns) {

    int idx = GETIDX();
    if (idx < nAtoms) {
        float4 posWhole = xsFrom[idx];
        uint id = idsFrom[idx];
        float3 pos = make_float3(posWhole);
        int3 sqrIdx = make_int3((pos - os) / ds);
        int sqrLinIdx = LINEARIDX(sqrIdx, ns);
        int sortedIdx = gridCellArrayIdxs[sqrLinIdx] + idxInGridCell[idx];
        //printf("I MOVE FROM %d TO %d, id is %d , MY POS IS %f %f %f\n", idx, sortedIdx, id, pos.x, pos.y, pos.z);

        //okay, now have all data needed to do copies
        copyToOtherList<float4>(xsFrom, xsTo, idx, sortedIdx);
        copyToOtherList<uint>(idsFrom, idsTo, idx, sortedIdx);
        copyToOtherList<float4>(vsFrom, vsTo, idx, sortedIdx);
        copyToOtherList<float4>(fsFrom, fsTo, idx, sortedIdx);
        copyToOtherList<float4>(fsLastFrom, fsLastTo, idx, sortedIdx);
        copyToOtherList<float>(qsFrom, qsTo, idx, sortedIdx);

        int xAddrId = XIDX(id, sizeof(int)) * sizeof(int);
        int yIdxId = YIDX(id, sizeof(int));

        surf2Dwrite(sortedIdx, idToIdx, xAddrId, yIdxId);

    //annnnd copied!


        




    }
}


__global__ void gridNonSort(float4 *xs, float4 *xsGrid, uint *ids, uint *idsGrid, int nAtoms, int *gridCellArrayIdxs, int *idxInGridCell, float3 os, float3 ds, int3 ns) {
    int idx = GETIDX();
    if (idx < nAtoms) {

        float4 posWhole = xs[idx];
        float3 pos = make_float3(posWhole);
        int3 sqrIdx = make_int3((pos - os) / ds);
        int sqrLinIdx = LINEARIDX(sqrIdx, ns); //only uses xyz
        int sortedIdx = gridCellArrayIdxs[sqrLinIdx] + idxInGridCell[idx];

        xsGrid[sortedIdx] = posWhole;
        idsGrid[sortedIdx] = ids[idx];

    }
}




__device__ void checkCell(float3 pos, int idx, uint myId, int myIdx, float4 *xs, uint *ids, int &myCount, int *gridCellArrayIdxs, cudaTextureObject_t idToIdxs, int squareIdx, float3 offset, float3 trace, float neighCutSqr) {
    int idxMin = gridCellArrayIdxs[squareIdx];
    int idxMax = gridCellArrayIdxs[squareIdx+1];
    float3 loop = offset * trace;
    for (int i=idxMin; i<idxMax; i++) {
        float4 otherPosWhole = xs[i];
        uint otherId = ids[i];
        float3 otherPos = make_float3(otherPosWhole);
        float3 distVec = otherPos + loop - pos;
        if (otherId != myId && dot(distVec, distVec) < neighCutSqr) {
            myCount ++;

        }

    }
}
__global__ void countNumNeighbors(float4 *xs, int nAtoms, cudaTextureObject_t idToIdxs, uint *ids, int *neighborCounts, int *gridCellArrayIdxs, float3 os, float3 ds, int3 ns, float3 periodic, float3 trace, float neighCutSqr, bool justSorted) {

    int idx = GETIDX();
    if (idx < nAtoms) {
        float4 posWhole = xs[idx];
        uint myId = ids[idx];



        float3 pos = make_float3(posWhole);
        int3 sqrIdx = make_int3((pos - os) / ds);
        int xIdx, yIdx, zIdx;
        int xIdxLoop, yIdxLoop, zIdxLoop;
        float3 offset = make_float3(0, 0, 0);
        int myIdx;
        if (justSorted) {
            myIdx = idx;
        } else {
            int xIdxID = XIDX(myId, sizeof(int));
            int yIdxID = YIDX(myId, sizeof(int));
            myIdx = tex2D<int>(idToIdxs, xIdxID, yIdxID);
        }

        int myCount = 0;
        for (xIdx=sqrIdx.x-1; xIdx<=sqrIdx.x+1; xIdx++) {
            offset.x = -floorf((float) xIdx / ns.x);
            xIdxLoop = xIdx + ns.x * offset.x;
       
            if (periodic.x || (!periodic.x && xIdxLoop == xIdx)) {

                for (yIdx=sqrIdx.y-1; yIdx<=sqrIdx.y+1; yIdx++) {
                    offset.y = -floorf((float) yIdx / ns.y);
                    yIdxLoop = yIdx + ns.y * offset.y;
                    if (periodic.y || (!periodic.y && yIdxLoop == yIdx)) {

                        for (zIdx=sqrIdx.z-1; zIdx<=sqrIdx.z+1; zIdx++) {
                            offset.z = -floorf((float) zIdx / ns.z);
                            zIdxLoop = zIdx + ns.z * offset.z;
                            if (periodic.z || (!periodic.z && zIdxLoop == zIdx)) {
                                int3 sqrIdxOther = make_int3(xIdxLoop, yIdxLoop, zIdxLoop);
                                int sqrIdxOtherLin = LINEARIDX(sqrIdxOther, ns);
                                checkCell(pos, idx, myId, myIdx, xs, ids, myCount, gridCellArrayIdxs, idToIdxs, sqrIdxOtherLin, -offset, trace, neighCutSqr);
                                //note sign switch on offset!

                            }
                        }
                    }
                }


            }
        }
        neighborCounts[idx] = myCount;
    }
}


__device__ uint addExclusion(uint otherId, uint *exclusionIds_shr, int idxLo, int idxHi) {
    uint exclMask = EXCL_MASK;
    for (int i=idxLo; i<idxHi; i++) {
        if ((exclusionIds_shr[i] & exclMask) == otherId) {
            return exclusionIds_shr[i] & (~exclMask);
        }
        
    }
    return 0;
}

__device__ int assignFromCell(float3 pos, int idx, uint myId, float4 *xs, uint *ids, int *nlistIdxs, int *gridCellArrayIdxs, cudaTextureObject_t idToIdxs, int squareIdx, float3 offset, float3 trace, float neighCutSqr, int currentNeighborIdx, uint *neighborlist, bool justSorted, uint *exclusionIds_shr, int exclIdxLo_shr, int exclIdxHi_shr, int warpSize) {
    uint idxMin = gridCellArrayIdxs[squareIdx];
    uint idxMax = gridCellArrayIdxs[squareIdx+1];
    for (uint i=idxMin; i<idxMax; i++) {
        float4 otherPosWhole = xs[i];
        float3 otherPos = make_float3(otherPosWhole);
        float3 distVec = otherPos + (offset * trace) - pos;
        uint otherId = ids[i];

        if (myId != otherId && dot(distVec, distVec) < neighCutSqr/* && !(isExcluded(otherId, exclusions, numExclusions, maxExclusions))*/) {
            uint exclusionTag = addExclusion(otherId, exclusionIds_shr, exclIdxLo_shr, exclIdxHi_shr);

            //if (myId==16) {
            //    printf("my id is 16 and my threadIdx is %d\n\n\n\n\n", threadIdx.x);
           // }
            if (justSorted) {


                neighborlist[currentNeighborIdx] = (i | exclusionTag);
            } else {
                int xIdxID = XIDX(otherId, sizeof(int));
                int yIdxID = YIDX(otherId, sizeof(int));
                uint otherIdx = tex2D<int>(idToIdxs, xIdxID, yIdxID);
                //if (myId==16) {
                //    printf("otherId is %d and idx is %d, looking up id from ids list gives %d\n", otherId, otherIdx, idsActive[otherIdx]);
               // }
                neighborlist[currentNeighborIdx] = (otherIdx | exclusionTag);

            }
            currentNeighborIdx += warpSize;
        }

    }
    return currentNeighborIdx;
}
__global__ void assignNeighbors(float4 *xs, int nAtoms, cudaTextureObject_t idToIdxs, uint *ids, int *nlistIdxs, int *gridCellArrayIdxs, int *cumulSumMaxPerBlock, float3 os, float3 ds, int3 ns, float3 periodic, float3 trace, float neighCutSqr, bool justSorted, uint *neighborlist, int warpSize, int *exclusionIndexes, uint *exclusionIds, int maxExclusionsPerAtom) {
  ///  extern __shared__ int exclusions_shr[]; 

    extern __shared__ uint exclusionIds_shr[];
    /*
    int tidLo = blockIdx.x * blockDim.x;
    int tidHi = min((blockIdx.x+1) * blockDim.x, nAtoms) - 1;
    int idLo = *(int *) &tex2D<float4>(xs, XIDX(tidLo, sizeof(float4)), YIDX(tidLo, sizeof(float4))).w;
    int idHi = *(int *) &tex2D<float4>(xs, XIDX(tidHi, sizeof(float4)), YIDX(tidHi, sizeof(float4))).w;
    int copyLo = exclusionIndexes[idLo];
    int copyHi = exclusionIndexes[idHi+1];

    copyToShared<uint>(exclusionIds + copyLo, exclusionIds_shr, copyHi - copyLo);
    __syncthreads();
    */
    //so the exclusions that this contiguous block of atoms needs are scattered around the exclusionIndexes list because they're sorted by id.  Need to copy it into shared.  Each thread has to copy from diff block b/c scatted
    int idx = GETIDX();
    float4 posWhole;
    int myId;
    int exclIdxLo_shr, exclIdxHi_shr, numExclusions;
    exclIdxLo_shr = threadIdx.x * maxExclusionsPerAtom;
    if (idx < nAtoms) {
        posWhole = xs[idx];
        myId = ids[idx];
        int exclIdxLo = exclusionIndexes[myId];
        int exclIdxHi = exclusionIndexes[myId+1];
        numExclusions = exclIdxHi - exclIdxLo;
        exclIdxHi_shr = exclIdxLo_shr + numExclusions;
        for (int i=exclIdxLo; i<exclIdxHi; i++) {
            uint exclusion = exclusionIds[i];
            exclusionIds_shr[maxExclusionsPerAtom*threadIdx.x + i - exclIdxLo] = exclusion;
            //printf("I am thread %d and I am copying %u from global %d to shared %d\n", threadIdx.x, exclusion, i, maxExclusionsPerAtom*threadIdx.x+i-exclIdxLo);
        }
    }
    //okay, now we have exclusions copied into shared
    __syncthreads();
    //int cumulSumUpToMe = cumulSumMaxPerBlock[blockIdx.x];
    //int maxNeighInMyBlock = cumulSumMaxPerBlock[blockIdx.x+1] - cumulSumUpToMe;
    //int myWarp = threadIdx.x / warpSize;
    //int myIdxInWarp = threadIdx.x % warpSize;
    //okay, then just start here and space by warpSize;
    //YOU JUST NEED TO UPDATE HOW WE CHECK EXCLUSIONS (IDXS IN SHEARED)
    if (idx < nAtoms) {
        //printf("threadid %d idx %x has lo, hi of %d, %d\n", threadIdx.x, idx, exclIdxLo_shr, exclIdxHi_shr);
        //not really template, figure out if can link externs later
        //HEY, so this is a problem, because it gets the index as if it were the index in the active list, BUT IF YOU DIDN'T SORT, THAT'S NOT TRUE.  So what you need to do is if you didn't sort, index = idxFromId, otherwise threadIdx
        int currentNeighborIdx;
        if (justSorted) {
            currentNeighborIdx = baseNeighlistIdx<void>(cumulSumMaxPerBlock, warpSize);
        } else {
            int xIdxID = XIDX(myId, sizeof(int));
            int yIdxID = YIDX(myId, sizeof(int));
            uint myIdx = tex2D<int>(idToIdxs, xIdxID, yIdxID);
            currentNeighborIdx = baseNeighlistIdxFromIndex<void>(cumulSumMaxPerBlock, warpSize, myIdx);

        }





        float3 pos = make_float3(posWhole);
        int3 sqrIdx = make_int3((pos - os) / ds);
        int xIdx, yIdx, zIdx;
        int xIdxLoop, yIdxLoop, zIdxLoop;
        float3 offset = make_float3(0, 0, 0);
       

        for (xIdx=sqrIdx.x-1; xIdx<=sqrIdx.x+1; xIdx++) {
            offset.x = -floorf((float) xIdx / ns.x);
            xIdxLoop = xIdx + ns.x * offset.x;
            if (periodic.x || (!periodic.x && xIdxLoop == xIdx)) {

                for (yIdx=sqrIdx.y-1; yIdx<=sqrIdx.y+1; yIdx++) {
                    offset.y = -floorf((float) yIdx / ns.y);
                    yIdxLoop = yIdx + ns.y * offset.y;
                    if (periodic.y || (!periodic.y && yIdxLoop == yIdx)) {

                        for (zIdx=sqrIdx.z-1; zIdx<=sqrIdx.z+1; zIdx++) {
                            offset.z = -floorf((float) zIdx / ns.z);
                            zIdxLoop = zIdx + ns.z * offset.z;
                            if (periodic.z || (!periodic.z && zIdxLoop == zIdx)) {
                                int3 sqrIdxOther = make_int3(xIdxLoop, yIdxLoop, zIdxLoop);
                                int sqrIdxOtherLin = LINEARIDX(sqrIdxOther, ns);

//__device__ int assignFromCell(float3 pos, int idx, uint myId, float4 *xs, uint *ids, int *nlistIdxs, int *gridCellArrayIdxs, cudaTextureObject_t idToIdxs, int squareIdx, float3 offset, float3 trace, float neighCutSqr, int currentNeighborIdx, cudaSurfaceObject_t neighborlist, bool justSorted, uint *exclusionIds_shr, int exclIdxLo_shr, int exclIdxHi_shr) {
                                currentNeighborIdx = assignFromCell(pos, idx, myId, xs, ids, nlistIdxs, gridCellArrayIdxs, idToIdxs, sqrIdxOtherLin, -offset, trace, neighCutSqr, currentNeighborIdx, neighborlist, justSorted, exclusionIds_shr, exclIdxLo_shr, exclIdxHi_shr, warpSize);

                            }
                        }
                    }
                }


            }
        }
    }
}

void GridGPU::initArrays() {
    perCellArray = GPUArray<int>(prod(ns) + 1);
    perAtomArray = GPUArray<int>(state->atoms.size()+1);
    perBlockArray = GPUArray<int>(NBLOCK(state->atoms.size()) + 1); //also cumulative sum, tracking cumul. sum of max per block
    xsLastBuild = GPUArrayDevice<float4>(state->atoms.size());
    //in prepare for run, you make GPU grid _after_ copying xs to device
    buildFlag = GPUArray<int>(1);
    buildFlag.d_data.memset(0);
}
void GridGPU::initStream() {
    //cout << "initializing stream" << endl;
    //streamCreated = true;
    //CUCHECK(cudaStreamCreate(&rebuildCheckStream));
}

GridGPU::GridGPU() {
    streamCreated = false;
    //initStream();
}
GridGPU::GridGPU(State *state_, float3 ds_, float3 dsOrig_, float3 os_, int3 ns_, float maxRCut_) : state(state_), ds(ds_), dsOrig(dsOrig_), os(os_), ns(ns_), neighCutoffMax(maxRCut_ + state->padding) {
    streamCreated = false;
    initArrays();
    initStream();
    handleExclusions();
    numChecksSinceLastBuild = 0;
};
GridGPU::GridGPU(State *state_, float dx_, float dy_, float dz_) : state(state_) {
    streamCreated = false;
	Vector trace = state->bounds.trace; //EEHHHHH SHOULD CHANGE TO BOUNDSGPU, but it doesn't really matter because you initialize them at the same time.  FOR NOW
	Vector attemptDDim = Vector(dx_, dy_, dz_);
	VectorInt nGrid = trace / attemptDDim; //so rounding to bigger grid
	Vector actualDDim = trace / nGrid; 
	//making grid that is exactly size of box.  This way can compute offsets easily from Grid that doesn't have to deal with higher-level stuff like bounds	
	is2d = state->is2d;
	ns = nGrid.asInt3();
	ds = actualDDim.asFloat3();
	os = state->boundsGPU.lo;
	if (is2d) {
		ns.z=1;
		ds.z=1;
		assert(os.z==-.5);
	}
	dsOrig = actualDDim.asFloat3();
    initArrays();
    initStream();
    handleExclusions();
    numChecksSinceLastBuild = 0;
};
GridGPU::~GridGPU() {
    if (streamCreated) {
        CUCHECK(cudaStreamDestroy(rebuildCheckStream));
    }
}

void GridGPU::copyPositionsAsync() {

    state->gpd.xs.d_data[state->gpd.activeIdx].copyToDeviceArray((void *) xsLastBuild.data());//, rebuildCheckStream);

}

/*
void printNeighborCounts(int *counts, int nAtoms) {
    cout << "neighbor counts" << endl;
    for (int i=0; i<nAtoms; i++) {
        cout << "n " <<  counts[i+1] - counts[i] << endl;
    }
    cout << "end" << endl;
}
*/
/*
__global__ void printStuff(int *vals, int n) {
    int idx = GETIDX();
    if (idx < n) {
        printf("%d: %d\n", idx, vals[idx]);
    }
}
*/
/*
void __global__ printBiz(float4 *xs, int n) {
    int idx = GETIDX();
    if (idx < n) {
        float4 x = xs[idx];
        printf("%d is %f %f %f\n", idx, x.x, x.y, x.z);

    }
}
*/
/*
void __global__ printBiz(cudaTextureObject_t tex, int n) {
    int idx = GETIDX();
    if (idx < n) {
        printf("%d is %d\n", idx, tex2D<int>(tex, XIDX(idx), YIDX(idx)));

    }
}


void __global__ printInts(int *xs, int n) {
    int idx = GETIDX();
    if (idx < n) {
        printf("%d is %d\n", idx, xs[idx]);
    }
}
*/
//nAtoms, neighborlist.surf, perAtomArray.ptr, exclusionIndexes.data(), exclusionIds.data(), state->gpd.xs.getTex(activeIdx));

/*
void __global__ addExclusions(int nAtoms, cudaSurfaceObject_t nlist, int *nlistIdxs, int *exclusionIndexes, uint *exclusionIds, cudaTextureObject_t xs) {
    uint exclusionMask = ~(3 << 30);
    extern __shared__ uint exclusionIds_shr[];
    int tidLo = blockIdx.x * blockDim.x;
    int tidHi = min((blockIdx.x+1) * blockDim.x, nAtoms) - 1;
    int idLo = *(int *) &tex2D<float4>(xs, XIDX(tidLo, sizeof(float4), YIDX(tidLo, sizeof(float4)))).w;
    int idHi = *(int *) &tex2D<float4>(xs, XIDX(tidHi, sizeof(float4), YIDX(tidHi, sizeof(float4)))).w;
    int copyLo = exclusionIndexes[idLo];
    int copyHi = exclusionIndexes[idHi+1];

    copyToShared<uint>(exclusionIds + copyLo, exclusionIds_shr, copyHi - copyLo);
    __syncthreads();
    //okay, now all of the exclusions are copied into shared
    int idx = GETIDX();
    if (idx < nAtoms) {
        int id = *(int *) &tex2D<float4>(xs, XIDX(idx, sizeof(float4), YIDX(idx, sizeof(float4)))).w;
        int idxLo_shr = exclusionIndexes[id] - copyLo;
        int idxHi_shr = exclusionIndexes[id+1] - copyLo;
        int nlistIdxLo = nlistIdxs[idx];
        int nlistIdxHi = nlistIdxs[idx+1];
        for (int nlistIdx=nlistIdxLo; nlistIdx<nlistIdxHi; nlistIdx++) {
            uint neighborId = tex2D<uint>(nlist, XIDX(nlistIdx, sizeof(uint)), YIDX(nlistIdx, sizeof(float4)));
            for (int i=idxLo_shr; i<idxHi_shr; i++) {
                uint excl = exclusionIds_shr[i];
                if (neighborId == (excl & exclusionMask)) {
                    uint dist = excl & (~exclusionMask);
                    neighborId |= dist;
                    surf2Dwrite(neighborId, nlist, xAddr, yIdx);
                    break;

                }
            }
        }


    }
        int3 sqrIdx = make_int3((make_float3(tex2D<float4>(xs, xIdx, yIdx)) - os) / ds);
    __device__ void copyToShared (T *src, T *dest, int n) {
}
*/

void setPerBlockCounts(vector<int> &neighborCounts, vector<int> &numNeighborsInBlocks) {
    numNeighborsInBlocks[0] = 0;
    for (int i=0; i<numNeighborsInBlocks.size()-1; i++) {
        int maxNeigh = 0;
        int maxIdx = fmin(neighborCounts.size()-1, (i+1)*PERBLOCK);
        for (int j=i*PERBLOCK; j<maxIdx; j++) {
            int numNeigh = neighborCounts[j];
            //cout << "summing at idx " << j << ", it has " << numNeigh << endl;
            maxNeigh = fmax(numNeigh, maxNeigh);
        }
        numNeighborsInBlocks[i+1] = numNeighborsInBlocks[i] + maxNeigh; //cumulative sum of # in block

    }
}

__global__ void setBuildFlag(float4 *xsA, float4 *xsB, int nAtoms, BoundsGPU boundsGPU, float paddingSqr, int *buildFlag, int numChecksSinceBuild) {
    int idx = GETIDX();
    extern __shared__ short flags_shr[];
    if (idx < nAtoms) {
        float3 distVector = boundsGPU.minImage(make_float3(xsA[idx] - xsB[idx]));
        float lenSqr = lengthSqr(distVector);
        float maxMoveRatio = fminf(0.95, (numChecksSinceBuild+1) / (float) (numChecksSinceBuild+2));
        float maxMoveSqr = paddingSqr * maxMoveRatio * maxMoveRatio;
        //printf("moved %f\n", sqrtf(lenSqr));
      //  printf("max move is %f\n", maxMoveSqr);
        flags_shr[threadIdx.x] = (short) (lenSqr > maxMoveSqr);
    } else {
        flags_shr[threadIdx.x] = 0;
    }
   __syncthreads();
   //just took from parallel reduction in cutils_func
   reduceByN<short>(flags_shr, blockDim.x);
    if (threadIdx.x == 0 and flags_shr[0] != 0) {
        buildFlag[0] = 1;
    }

}
void GridGPU::periodicBoundaryConditions(float neighCut, bool doSort, bool forceBuild) {
    //to do: remove sorting option.  Must sort every time if using mpi, and also I think building without sorting isn't even working right now
    if (neighCut == -1) {
        neighCut = neighCutoffMax;
    }

    int warpSize = state->devManager.prop.warpSize;
    Vector nsV = Vector(make_float3(ns));
    int nAtoms = state->atoms.size();
    int activeIdx = state->gpd.activeIdx;
    BoundsGPU bounds = state->boundsGPU;
    //DO ASYNC COPY TO xsLastBuild
    //FINISH FUTURE WHICH SETS REBUILD FLAG BY NOW PLEASE
   // CUCHECK(cudaStreamSynchronize(rebuildCheckStream));
    setBuildFlag<<<NBLOCK(nAtoms), PERBLOCK, PERBLOCK * sizeof(short)>>>(state->gpd.xs(activeIdx), xsLastBuild.data(), nAtoms, bounds, state->padding*state->padding, buildFlag.d_data.data(), numChecksSinceLastBuild);
    buildFlag.dataToHost();
    cudaDeviceSynchronize();
    //    cout << "I AM BUILDING" << endl;
    if (buildFlag.h_data[0] or forceBuild) {

        float3 ds_orig = ds;
        float3 os_orig = os;
        ds += make_float3(EPSILON, EPSILON, EPSILON); //as defined in Vector.h.  PAIN AND NUMERICAL ERROR AWAIT ALL THOSE WHO ALTER THIS LINE (AND THE ONE BELOW IT)
        os -= make_float3(EPSILON, EPSILON, EPSILON);
        BoundsGPU boundsUnskewed = bounds.unskewed();
        float3 trace = boundsUnskewed.trace();
        if (bounds.sides[0].y or bounds.sides[1].x) {
            Mod::unskewAtoms<<<NBLOCK(nAtoms), PERBLOCK>>>(state->gpd.xs(activeIdx), nAtoms, bounds.sides[0], bounds.sides[1], bounds.lo);
        }
        periodicWrap<<<NBLOCK(nAtoms), PERBLOCK>>>(state->gpd.xs(activeIdx), nAtoms, boundsUnskewed);
        //SAFECALL((periodicWrap<<<NBLOCK(nAtoms), PERBLOCK>>>(state->gpd.xs(activeIdx), nAtoms, boundsUnskewed)), "wrap");
     //   float4 *xs = state->gpd.xs.getDevData();
        //cout << "I am here to help" << endl;
        //printBiz<<<NBLOCK(state->atoms.size()), PERBLOCK>>>(state->gpd.xs(activeIdx), state->atoms.size());
      //  cout << state->gpd.xs.size << endl;
      //  cudaDeviceSynchronize();
        //for (int i=0; i<state->atoms.size(); i++) {
        //    cout << Vector(xs[i]) << endl;
        //}
        int numGridCells = prod(ns);
        if (numGridCells + 1 != perCellArray.size()) {
            perCellArray = GPUArray<int>(numGridCells + 1);
        }
        perCellArray.d_data.memset(0);
        perAtomArray.d_data.memset(0);
      //  cudaDeviceSynchronize();
        countNumInGridCells<<<NBLOCK(nAtoms), PERBLOCK>>>(state->gpd.xs(activeIdx), nAtoms, perCellArray.d_data.data(), perAtomArray.d_data.data(), os, ds, ns);
        //SAFECALL((countNumInGridCells<<<NBLOCK(nAtoms), PERBLOCK>>>(state->gpd.xs(activeIdx), nAtoms, perCellArray.d_data.data(), perAtomArray.d_data.data(), os, ds, ns)), "NUM IN CELLS");
        //countNumInGridCells<<<NBLOCK(nAtoms), PERBLOCK>>>(state->gpd.xs(activeIdx), nAtoms, perCellArray.d_data.data(), perAtomArray.d_data.data(), os, ds, ns);
        perCellArray.dataToHost();
        cudaDeviceSynchronize();
        int *gridCellCounts_h = perCellArray.h_data.data();
        
        cumulativeSum(gridCellCounts_h, perCellArray.size());//repurposing this as starting indexes for each grid square
        perCellArray.dataToDevice();
        int gridIdx;
        if (doSort) {
            sortPerAtomArrays<<<NBLOCK(nAtoms), PERBLOCK>>>(

                    state->gpd.xs(activeIdx),  
                    state->gpd.xs(!activeIdx),

                    state->gpd.vs(activeIdx),
                    state->gpd.vs(!activeIdx),

                    state->gpd.fs(activeIdx),
                    state->gpd.fs(!activeIdx),

                    state->gpd.fsLast(activeIdx),
                    state->gpd.fsLast(!activeIdx),

                    state->gpd.ids(activeIdx),
                    state->gpd.ids(!activeIdx),

                    state->gpd.qs(activeIdx),
                    state->gpd.qs(!activeIdx),

                    state->gpd.idToIdxs.getSurf(),

                    perCellArray.d_data.data(), perAtomArray.d_data.data(), nAtoms, os, ds, ns
                    );
            activeIdx = state->gpd.switchIdx();
            gridIdx = activeIdx;
        } else { //otherwise, just use non-active xs array as grid storage
            //gridCPU(state->gpd.xs, activeIdx, nAtoms, perCellArray, perAtomArray, os, ds, ns);
            gridNonSort<<<NBLOCK(nAtoms), PERBLOCK>>>(state->gpd.xs(activeIdx), state->gpd.xs(!activeIdx), state->gpd.ids(activeIdx), state->gpd.ids(!activeIdx), nAtoms, perCellArray.d_data.data(), perAtomArray.d_data.data(), os, ds, ns);
            gridIdx = !activeIdx;

        }

        perAtomArray.d_data.memset(0);
        //SAFECALL((countNumNeighbors<<<NBLOCK(nAtoms), PERBLOCK>>>(state->gpd.xs(gridIdx), nAtoms, state->gpd.idToIdxs.getTex(), state->gpd.ids(gridIdx), perAtomArray.d_data.data(), perCellArray.d_data.data(), os, ds, ns, bounds.periodic, trace, neighCut*neighCut, doSort)), "NUM NEIGH");
        countNumNeighbors<<<NBLOCK(nAtoms), PERBLOCK>>>(state->gpd.xs(gridIdx), nAtoms, state->gpd.idToIdxs.getTex(), state->gpd.ids(gridIdx), perAtomArray.d_data.data(), perCellArray.d_data.data(), os, ds, ns, bounds.periodic, trace, neighCut*neighCut, doSort);//, state->gpd.nlistExclusionIdxs.getTex(), state->gpd.nlistExclusions.getTex(), state->maxExclusions);
        perAtomArray.dataToHost();
        cudaDeviceSynchronize();
        
        setPerBlockCounts(perAtomArray.h_data, perBlockArray.h_data);  //okay, now this is the start index (+1 is end index) of each atom's neighbors
        perBlockArray.dataToDevice();

        int totalNumNeighbors = perBlockArray.h_data.back() * PERBLOCK;
        //cout << "TOTAL NUM IS " << totalNumNeighbors << endl;
        if (totalNumNeighbors > neighborlist.size()) {
            neighborlist = GPUArrayDevice<uint>(totalNumNeighbors*1.5);
        } else if (totalNumNeighbors < neighborlist.size() * 0.5) {
            neighborlist = GPUArrayDevice<uint>(totalNumNeighbors*0.8);
        }
        /*
        SAFECALL((assignNeighbors<<<NBLOCK(nAtoms), PERBLOCK, PERBLOCK*maxExclusionsPerAtom*sizeof(uint)>>>(
                state->gpd.xs(gridIdx), 
                nAtoms, 
                state->gpd.idToIdxs.getTex(), 
                state->gpd.ids(gridIdx),
                perAtomArray.d_data.data(),
                perCellArray.d_data.data(),
                perBlockArray.d_data.data(),
                os, ds, ns, bounds.periodic, trace, neighCut*neighCut, doSort, neighborlist.data(), warpSize,
                exclusionIndexes.data(), exclusionIds.size(), maxExclusionsPerAtom
                
                )), "ASSIGN");//, state->gpd.nlistExclusionIdxs.getTex(), state->gpd.nlistExclusions.getTex(), state->maxExclusions);

        
                */
        assignNeighbors<<<NBLOCK(nAtoms), PERBLOCK, PERBLOCK*maxExclusionsPerAtom*sizeof(uint)>>>(
                state->gpd.xs(gridIdx), 
                nAtoms, 
                state->gpd.idToIdxs.getTex(), 
                state->gpd.ids(gridIdx),
                perAtomArray.d_data.data(),
                perCellArray.d_data.data(),
                perBlockArray.d_data.data(),
                os, ds, ns, bounds.periodic, trace, neighCut*neighCut, doSort, neighborlist.data(), warpSize,
                exclusionIndexes.data(), exclusionIds.data(), maxExclusionsPerAtom
                
                );//, state->gpd.nlistExclusionIdxs.getTex(), state->gpd.nlistExclusions.getTex(), state->maxExclusions);
                


        //printNeighbors<<<NBLOCK(state->atoms.size()), PERBLOCK>>>(perAtomArray.ptr, neighborlist.tex, state->atoms.size());
        /*
        int *neighCounts = perAtomArray.get((int *) NULL);
        cudaDeviceSynchronize();
       printNeighborCounts(neighCounts, state->atoms.size());
       free(neighCounts);
       */
        if (bounds.sides[0].y or bounds.sides[1].x) {
            Mod::skewAtomsFromZero<<<NBLOCK(nAtoms), PERBLOCK>>>(state->gpd.xs(activeIdx), nAtoms, bounds.sides[0], bounds.sides[1], bounds.lo);
        }
        ds = ds_orig;
        os = os_orig;
        //verifyNeighborlists(neighCut);

        numChecksSinceLastBuild = 0; 
        copyPositionsAsync();
    } else {
        numChecksSinceLastBuild++;
    }
    buildFlag.d_data.memset(0); 
    
}





bool GridGPU::verifyNeighborlists(float neighCut) {
    cout << "going to verify" << endl;
    uint *nlist = neighborlist.get((uint *) NULL);
    float cutSqr = neighCut * neighCut;
    perAtomArray.dataToHost();
    int *neighCounts = perAtomArray.h_data.data();
    state->gpd.xs.dataToHost();
    state->gpd.ids.dataToHost();
    cudaDeviceSynchronize();
 //   cout << "Neighborlist" << endl;
  //  for (int i=0; i<neighborlist.size(); i++) {
  //      cout << "idx " << i << " " << nlist[i] << endl;
  //  }
  //  cout << "end neighborlist" << endl;
    vector<float4> xs = state->gpd.xs.h_data;
    vector<uint> ids = state->gpd.ids.h_data;
  //  cout << "ids" << endl;
 //  for (int i=0; i<ids.size(); i++) {
 //       cout << ids[i] << endl;
 //   }
    state->gpd.xs.dataToHost(!state->gpd.xs.activeIdx);
    cudaDeviceSynchronize();
    vector<float4> sortedXs = state->gpd.xs.h_data;
  //  int gpuId = *(int *)&sortedXs[TESTIDX].w;

//    int cpuIdx = gpuId;
    vector<vector<int> > cpu_neighbors;
    for (int i=0; i<xs.size(); i++) {
        vector<int> atom_neighbors;
        float3 self = make_float3(xs[i]);
        for (int j=0; j<xs.size(); j++) {
            if (i!=j) {
                float4 otherWhole = xs[j];
                float3 minImage = state->boundsGPU.minImage(self - make_float3(otherWhole));
                if (lengthSqr(minImage) < cutSqr) {
                    uint otherId = ids[j];
                    atom_neighbors.push_back(otherId);
                }

            }
        }
        sort(atom_neighbors.begin(), atom_neighbors.end());
        cpu_neighbors.push_back(atom_neighbors);
    }
//    cout << "cpu dist is " << sqrt(lengthSqr(state->boundsGPU.minImage(xs[0]-xs[1])))  << endl;
    int warpSize = state->devManager.prop.warpSize;
    for (int i=0; i<xs.size(); i++) {
        int blockIdx = i / PERBLOCK;
        int warpIdx = (i - blockIdx * PERBLOCK) / warpSize;
        int idxInWarp = i - blockIdx * PERBLOCK - warpIdx * warpSize;
        int cumSumUpToMyBlock = perBlockArray.h_data[blockIdx];
        int perAtomMyWarp = perBlockArray.h_data[blockIdx+1] - cumSumUpToMyBlock;
        int baseIdx = PERBLOCK * perBlockArray.h_data[blockIdx] + perAtomMyWarp * warpSize * warpIdx + idxInWarp;
        //cout << "i is " << i << " blockIdx is " << blockIdx << " warp idx is " << warpIdx << " and idx in that warp is " << idxInWarp << " resulting base idx is " << baseIdx << endl;
        //cout << "id is " << ids[i] << endl;
        vector<int> neighIds;
    //    cout << "begin end " << neighIdxs[i] << " " << neighIdxs[i+1] << endl;
        for (int j=0; j<neighCounts[i]; j++) {
            int nIdx = baseIdx + j*warpSize;
            //cout << "looking at neighborlist index " << nIdx << endl;

      //      cout << "idx " << nlist[nIdx] << endl;
            float4 atom = xs[nlist[nIdx]];
            uint id = ids[nlist[nIdx]];
       //     cout << "id is " << id << endl;
            neighIds.push_back(id);
        }
        sort(neighIds.begin(), neighIds.end());
        if (neighIds != cpu_neighbors[i]) {
            cout << "problem at idx " << i << " id " << ids[i] << endl;
            cout << "cpu " << cpu_neighbors[i].size() << " gpu " << neighIds.size() << endl;
            cout << "cpu neighbor ids" << endl;
            for (int x : cpu_neighbors[i]) {
                cout << x << " ";
            }
            cout << endl;
            cout << "gpu neighbor ids" << endl;
            for (int x : neighIds) {
                cout << x << " ";
            }
            cout << endl;
            break;

        }

    }
    /*
    bool pass = true;
    for (int i=0; i<xs.size(); i++) {
        if (nneigh[i] != cpu_check[i]) {
            vector<int> gpuIdxs, cpuIdxs;
            for (int listIdx=neighIdxs[i]; listIdx < neighIdxs[i+1]; listIdx++) {
                gpuIdxs.push_back(nlist[listIdx]);
            }
            for (int j=0; j<xs.size(); j++) {
                if (i!=j) {
                    float3 minImage = state->boundsGPU.minImage(xs[i] - xs[j]);
                    if (lengthSqr(minImage) < cutSqr) {
                        cpuIdxs.push_back(j);
                    }

                }
            }
            for (int nIdx : gpuIdxs) {
                if (find(cpuIdxs.begin(), cpuIdxs.end(), nIdx) == cpuIdxs.end()) {
                    cout << "cpu is missing neighbor with dist " << length(state->boundsGPU.minImage(xs[i]-xs[nIdx])) << endl;
                    cout << Vector(xs[i]) << "      " << Vector(xs[nIdx]) << "    " << nIdx << endl;
                }
            }
            for (int nIdx : cpuIdxs) {
                if (find(gpuIdxs.begin(), gpuIdxs.end(), nIdx) == gpuIdxs.end()) {
                    cout << "gpu is missing neighbor with dist " << length(state->boundsGPU.minImage(xs[i]-xs[nIdx])) << endl;
                    cout << Vector(xs[i]) << "      " << Vector(xs[nIdx]) << "    " << nIdx << endl;
                }
            }

            cout << nneigh[i] << " on gpu " << cpu_check[i] << " on cpu " << endl;
            //cout << Vector(xs[i]) << endl;
            pass = false;
        }
    }
    if (pass) {
    //    cout << "neighbor count passed" << endl;
    }
    */
    free(nlist);
    cout << "end verification" << endl;
    return true;

}
bool GridGPU::checkSorting(int gridIdx, int *gridIdxs, GPUArrayDevice<int> &gridIdxsDev) {
   // printInts<<<NBLOCK(gridIdxsDev.n), PERBLOCK>>>(gridIdxsDev.ptr, gridIdxsDev.n);
    int numGridIdxs = prod(ns);
    vector<int> activeIds = LISTMAPREF(Atom, int, atom, state->atoms, atom.id);
    vector<int> gpuIds;

    gpuIds.reserve(activeIds.size());
    state->gpd.xs.dataToHost(gridIdx);
    cudaDeviceSynchronize();
    vector<float4> &xs = state->gpd.xs.h_data;
    bool correct = true;
    for (int i=0; i<numGridIdxs; i++) {
        int gridLo = gridIdxs[i];
        int gridHi = gridIdxs[i+1];
     //   cout << "hi for " << i << " is " << gridHi << endl;
        for (int atomIdx=gridLo; atomIdx<gridHi; atomIdx++) {
            float4 posWhole = xs[atomIdx];
            float3 pos = make_float3(posWhole);
            int id = *(int *) &posWhole.w;
            gpuIds.push_back(id);
            int3 sqr = make_int3((pos - os) / ds);
            int linear = LINEARIDX(sqr, ns);
            if (linear != i) {
                correct = false;
            }
        }
    }
    sort(activeIds.begin(), activeIds.end());
    sort(gpuIds.begin(), gpuIds.end());
    cout << activeIds.size() << " " << gpuIds.size() << endl;
    if (activeIds != gpuIds) {
        correct = false;
        cout << "different ids!   Seriou problem!" << endl;
        assert(activeIds.size() == gpuIds.size());
    }
    return correct;


    
}


void GridGPU::handleExclusions() {
    
    const ExclusionList exclList = generateExclusionList(4);
    vector<int> idxs;
    vector<uint> excludedById;
    excludedById.reserve(state->maxIdExisting+1);
    
    auto fillToId = [&] (int id) { //paired list is indexed by id.  Some ids could be missing, so need to fill in empty values
        while (idxs.size() <= id) {
            idxs.push_back(excludedById.size());
        }
    };

    uint exclusionTags[3] = {(uint) 1 << 30, (uint) 2 << 30, (uint) 3 << 30};
    maxExclusionsPerAtom = 0;
    for (auto it = exclList.begin(); it!=exclList.end(); it++) { //is ordered map, so it sorted by ascending id
        int id = it->first;
        //cout << "id is " << id << endl;
        const vector<set<int> > &atomExclusions = it->second;
        fillToId(id); 
        //cout << "filled" << endl;
        //for (int id : idxs) {
        //    cout << id << endl;
        //}
        for (int i=0; i<atomExclusions.size(); i++) {
            const set<int> &idsAtLevel = atomExclusions[i];
            for (auto itId=idsAtLevel.begin(); itId!=idsAtLevel.end(); itId++) {
                uint id = *itId;
                id |= exclusionTags[i];
                excludedById.push_back(id);


            }
        }
        idxs.push_back(excludedById.size());
        maxExclusionsPerAtom = fmax(maxExclusionsPerAtom, idxs.back() - idxs[idxs.size()-2]);
    }
  //  cout << "max excl per atom is " << maxExclusionsPerAtom << endl;
    exclusionIndexes = GPUArrayDevice<int>(idxs.size());
    exclusionIndexes.set(idxs.data());
    exclusionIds = GPUArrayDevice<uint>(excludedById.size());
    exclusionIds.set(excludedById.data());
    //atoms is sorted by id.  list of ids may be sparse, so need to make sure there's enough shared memory for PERBLOCK _atoms_, not just PERBLOCK ids (when calling assign exclusions kernel)

       //for test output
    /*
    cout << "index ptrs " << endl;
    for (int id : idxs) {
        cout << id << endl;
    }
    cout << "end" << endl;
    for (int i=0; i<idxs.size()-1; i++) {
        for (int exclIdx=idxs[i]; exclIdx < idxs[i+1]; exclIdx++) {
            uint excl = excludedById[exclIdx];
            uint filter = (uint) 3 << 30;
            cout << filter << endl;
            uint dist = (excl & filter) >> 30;
            uint id = excl & (~filter);
            cout << "id " << i << " excludes " << id << " with dist " << dist << endl;
        }
    }
    */
}




bool GridGPU::closerThan(const ExclusionList &exclude, 
                       int atomid, int otherid, int16_t depthi) {
    bool closerThan = false;
    // because we want to check lower depths
    --depthi;
    while (depthi >= 0) {
        const set<int> &closer = exclude.at(atomid)[depthi];
        closerThan |= (closer.find(otherid) != closer.end());
        --depthi;
    }
    // atoms are closer to themselves than any other depth away
    closerThan |= (atomid == otherid);
    return closerThan;
}

// allows us to extract any type of Bond from a BondVariant
class bondDowncast : public boost::static_visitor<const Bond &> {
	const BondVariant &_bv;
	public:
		bondDowncast(BondVariant &bv) : _bv(bv) {}
		template <typename T>
		const Bond &operator()(const T &b) const {
			return boost::get<T>(_bv);
		}
};

GridGPU::ExclusionList GridGPU::generateExclusionList(const int16_t maxDepth) {
    
    ExclusionList exclude;
    // not called depth because it's really the depth index, which is one
    // smaller than the depth
    int16_t depthi = 0;
    
    // computes adjacent bonds (depth -> 1, depthi -> 0)
    vector<vector<BondVariant> *> allBonds;
    for (Fix *f : state->fixes) {
        vector<BondVariant> *fixBonds = f->getBonds();
        if (fixBonds != nullptr) {
            allBonds.push_back(fixBonds);
        }
    }
    for (Atom atom : state->atoms) {
        exclude[atom.id].push_back(set<int>());
    }

		//typedef map<int, vector<set<int>>> ExclusionList;
    for (vector<BondVariant> *fixBonds : allBonds) {
        for (BondVariant &bondVariant : *fixBonds) {
			// boost variant magic that takes any BondVariant and turns it into a Bond
            const Bond &bond = boost::apply_visitor(bondDowncast(bondVariant), bondVariant);
            // atoms in the same bond are 1 away from each other
            exclude[bond.getAtomId(0)][depthi].insert(bond.getAtomId(1));
            exclude[bond.getAtomId(1)][depthi].insert(bond.getAtomId(0));
        }
    }
    depthi++;
    
    // compute the rest
    while (depthi < maxDepth) {
        for (Atom atom : state->atoms) {
            // for every atom at the previous depth away
            exclude[atom.id].push_back(set<int>());
            for (int extendFrom : exclude[atom.id][depthi-1]) {
                // extend to all atoms bonded with it
                exclude[atom.id][depthi].insert(
                  exclude[extendFrom][0].begin(), exclude[extendFrom][0].end());
            }
            // remove all atoms that are already excluded to a lower degree
            // TODO: may be a more efficient way
            for (auto it = exclude[atom.id][depthi].begin();
                 it != exclude[atom.id][depthi].end(); /*blank*/ ) {
                if (closerThan(exclude, atom.id, *it, depthi)) {
                   exclude[atom.id][depthi].erase(it++);
                } else {
                    ++it;
                }
            }
        }
        depthi++;
    }
    return exclude;
}

