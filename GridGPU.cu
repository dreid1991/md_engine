#include "GridGPU.h"
#include <set>
#include "State.h"
#include "helpers.h"
//for debugging
__global__ void countNumInGridCells(cudaTextureObject_t xs, int nAtoms, int *counts, int *atomIdxs, float3 os, float3 ds, int3 ns) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        //printf("idx %d\n", idx);
        int xIdx = XIDX(idx, sizeof(float4));
        int yIdx = YIDX(idx, sizeof(float4));
        int3 sqrIdx = make_int3((make_float3(tex2D<float4>(xs, xIdx, yIdx)) - os) / ds);
        int sqrLinIdx = LINEARIDX(sqrIdx, ns);
        //printf("lin is %d\n", sqrLinIdx);
        int myPlaceInGrid = atomicAdd(counts + sqrLinIdx, 1); //atomicAdd returns old value
        //printf("grid is %d\n", myPlaceInGrid);
        //printf("myPlaceInGrid %d\n", myPlaceInGrid);
        atomIdxs[idx] = myPlaceInGrid;
        //okay - atoms seem to be getting assigned the right idx in grid 
    }
}


__global__ void periodicWrap(cudaSurfaceObject_t xs, int nAtoms, BoundsGPU bounds) {
    int idx = GETIDX();
    if (idx < nAtoms) {

        int xIdx = XIDX(idx, sizeof(float4));
        int yIdx = YIDX(idx, sizeof(float4));
        int xAddr = xIdx * sizeof(float4);
        float4 pos = surf2Dread<float4>(xs, xAddr, yIdx);
        float4 orig = pos;
        float id = pos.w;
        float3 trace = bounds.trace();
        float3 diffFromLo = make_float3(pos) - bounds.lo;
        float3 imgs = floorf(diffFromLo / trace); //are unskewed at this point
        float3 pos_orig = make_float3(pos);
        pos -= make_float4(trace * imgs * bounds.periodic);
        pos.w = id;
        if (not(pos.x==orig.x and pos.y==orig.y and pos.z==orig.z)) { //sigh
            surf2Dwrite(pos, xs, xAddr, yIdx);
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
        cudaSurfaceObject_t xsFrom, cudaSurfaceObject_t xsTo, 
        float4  *vsFrom,        float4  *vsTo,
        float4  *fsFrom,        float4  *fsTo,
        float4  *fsLastFrom,    float4  *fsLastTo,
        cudaSurfaceObject_t typesFrom, cudaSurfaceObject_t typesTo,
        cudaSurfaceObject_t qsFrom, cudaSurfaceObject_t qsTo,

        cudaSurfaceObject_t idToIdx,
        int *gridCellArrayIdxs, int *idxInGridCell, int nAtoms, float3 os, float3 ds, int3 ns) {

    int idx = GETIDX();
    if (idx < nAtoms) {
        int xIdx = XIDX(idx, sizeof(float4));
        int yIdx = YIDX(idx, sizeof(float4));
        int xAddr = xIdx * sizeof(float4);
        float4 posWhole = surf2Dread<float4>(xsFrom, xAddr, yIdx);
        int id = * (int *) &posWhole.w;
        float3 pos = make_float3(posWhole);
        int3 sqrIdx = make_int3((pos - os) / ds);
        int sqrLinIdx = LINEARIDX(sqrIdx, ns);
        int sortedIdx = gridCellArrayIdxs[sqrLinIdx] + idxInGridCell[idx];

        //okay, now have all data needed to do copies
        copyToOtherSurf<float4>(xsFrom, xsTo, idx, sortedIdx);
        copyToOtherSurf<short>(typesFrom, typesTo, idx, sortedIdx);
        copyToOtherList<float4>(vsFrom, vsTo, idx, sortedIdx);
        copyToOtherList<float4>(fsFrom, fsTo, idx, sortedIdx);
        copyToOtherList<float4>(fsLastFrom, fsLastTo, idx, sortedIdx);
        copyToOtherSurf<float>(qsFrom, qsTo, idx, sortedIdx);

        int xAddrId = XIDX(id, sizeof(int)) * sizeof(int);
        int yIdxId = YIDX(id, sizeof(int));

        surf2Dwrite(sortedIdx, idToIdx, xAddrId, yIdxId);

    //annnnd copied!


        




    }
}


//        gridNonSort<<<NBLOCK(nAtoms), PERBLOCK>>>(state->gpd.xs.tex[activeIdx], state->gpd.xs.surf[!activeIdx], state->gpd.ids(activeIdx), state->gpd.ids(!activeIdx), nAtoms, perCellArray.ptr, perAtomArray.ptr, os, ds, ns);
__global__ void gridNonSort(cudaTextureObject_t xs, cudaSurfaceObject_t xsGrid, int nAtoms, int *gridCellArrayIdxs, int *idxInGridCell, float3 os, float3 ds, int3 ns) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        int xIdx = XIDX(idx, sizeof(float4));
        int yIdx = YIDX(idx, sizeof(float4));
        float4 posWhole = tex2D<float4>(xs, xIdx, yIdx);
        float3 pos = make_float3(posWhole);
        int3 sqrIdx = make_int3((pos - os) / ds);
        int sqrLinIdx = LINEARIDX(sqrIdx, ns); //only uses xyz
        int sortedIdx = gridCellArrayIdxs[sqrLinIdx] + idxInGridCell[idx];

        xIdx = XIDX(sortedIdx, sizeof(float4));
        yIdx = YIDX(sortedIdx, sizeof(float4));
        int xAddr = xIdx * sizeof(float4);
        surf2Dwrite(posWhole, xsGrid, xAddr, yIdx); //id is carried along with this

    }
}
/*
void gridCPU(GPUArrayTexPair<float4> &xs, int activeIdx, int nAtoms, GPUArrayDevice<int> &perCellArray, GPUArrayDevice<int> &perAtomArray, float3 os, float3 ds, int3 ns) {

    set<int> sortedAtoms;
    xs.dataToHost();
    int *gridCellArrayIdxs = perCellArray.get((int *) NULL);
    int *idxInGridCell = perAtomArray.get((int *) NULL);
    cudaDeviceSynchronize();
    for (int i=0; i<nAtoms; i++) {
        float4 posWhole = xs.h_data[i];
        float3 pos = make_float3(posWhole);
        if (pos.x < os.x or pos.y < os.y or pos.z < os.z or pos.x>=os.x+(ds.x*ns.x) or pos.y>=os.y+(ds.y*ns.y) or pos.z>=os.z+(ds.z*ns.z)) {
            if (fabs(pos.x) != 0 and fabs(pos.y) != 0) {
                cout << fabs
                cout << "Bad position " << Vector(pos) << endl;
            }
        }
        int3 sqrIdx = make_int3((pos-os)/ds);
        int sqrLinIdx = LINEARIDX(sqrIdx, ns);
        if (sqrLinIdx < 0 or sqrLinIdx >= perCellArray.n) {
            cout << "bad cell array index " << sqrLinIdx << endl;
            cout << "my pos is " << Vector(pos) << endl;
        }
        int sortedIdx = gridCellArrayIdxs[sqrLinIdx] + idxInGridCell[i];
        if (sortedIdx < 0 or sortedIdx >= nAtoms) {
            cout << "sorted index out of bounds!" << endl;
        }
        auto inserted = sortedAtoms.insert(sortedIdx);
        if (!inserted.second) {
            cout << "duplicate index!" << endl;
        }

    }
    free(gridCellArrayIdxs);
    free(idxInGridCell);

}
*/

__device__ bool isExcluded(const int id, int *exclusions, const int numExclusions, const int maxExclusions) { //exclusions should be shared memory or this will be just silly-slow
    for (int i=0; i<numExclusions; i++) {
        if (id == exclusions[maxExclusions * threadIdx.x + i]) {
            return true;
        }
    }
    return false;
}

__device__ void checkCell(float3 pos, int idx, int myId, int myIdx, cudaTextureObject_t xs, int *neighborCounts, int *gridCellArrayIdxs, cudaTextureObject_t idToIdxs, int squareIdx, float3 offset, float3 trace, float neighCutSqr) {//, int *exclusions, int numExclusions, int maxExclusions) {
    int idxMin = gridCellArrayIdxs[squareIdx];
    int idxMax = gridCellArrayIdxs[squareIdx+1];
    float3 loop = offset * trace;
    for (int i=idxMin; i<idxMax; i++) {
        int xIdx = XIDX(i, sizeof(float4));
        int yIdx = YIDX(i, sizeof(float4));
        float4 otherPosWhole = tex2D<float4>(xs, xIdx, yIdx); 
        float3 otherPos = make_float3(otherPosWhole);
        float3 distVec = otherPos + loop - pos;
        int otherId = *(int *) &otherPosWhole.w;
        if (otherId != myId && dot(distVec, distVec) < neighCutSqr /*&& !(isExcluded(otherId, exclusions, numExclusions, maxExclusions))*/) {
            neighborCounts[myIdx] ++;

        }

    }
}
__global__ void countNumNeighbors(cudaTextureObject_t xs, int nAtoms, cudaTextureObject_t idToIdxs, int *neighborCounts, int *gridCellArrayIdxs, float3 os, float3 ds, int3 ns, float3 periodic, float3 trace, float neighCutSqr, bool justSorted/*, cudaTextureObject_t exclusionIdxs, cudaTextureObject_t exclusions, int maxExclusions*/) {

   // extern __shared__ int exclusions_shr[]; 
    int idx = GETIDX();
    if (idx < nAtoms) {
        int xIdxAtom = XIDX(idx, sizeof(float4));
        int yIdxAtom = YIDX(idx, sizeof(float4));
        float4 posWhole = tex2D<float4>(xs, xIdxAtom, yIdxAtom);
        int myId = *(int *)&posWhole.w;


        /*int exclIdxLo = tex2D<int>(exclusionIdxs, XIDX(myId, sizeof(int)), YIDX(myId, sizeof(int)));
        int exclIdxHi = tex2D<int>(exclusionIdxs, XIDX(myId+1, sizeof(int)), YIDX(myId+1, sizeof(int)));
        int numExclusions = exclIdxHi - exclIdxLo;
        for (int i=0; i<numExclusions; i++) {
            exclusions_shr[threadIdx.x*maxExclusions + i] = tex2D<int>(exclusions, XIDX(exclIdxLo + i, sizeof(int)), YIDX(exclIdxLo + i, sizeof(int)));
        }
*/
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
                                checkCell(pos, idx, myId, myIdx, xs, neighborCounts, gridCellArrayIdxs, idToIdxs, sqrIdxOtherLin, -offset, trace, neighCutSqr);//, exclusions_shr, numExclusions, maxExclusions);
                                //note sign switch on offset!

                            }
                        }
                    }
                }


            }
        }
    }
}

__device__ int assignFromCell(float3 pos, int idx, int myId, cudaTextureObject_t xs, int *nlistIdxs, int *gridCellArrayIdxs, cudaTextureObject_t idToIdxs, int squareIdx, float3 offset, float3 trace, float neighCutSqr, int currentNeighborIdx, cudaSurfaceObject_t neighborlist, bool justSorted) {//, int *exclusions, int numExclusions, int maxExclusions) {
    int idxMin = gridCellArrayIdxs[squareIdx];
    int idxMax = gridCellArrayIdxs[squareIdx+1];
    for (int i=idxMin; i<idxMax; i++) {
        int xIdx = XIDX(i, sizeof(float4));
        int yIdx = YIDX(i, sizeof(float4));
        float4 otherPosWhole = tex2D<float4>(xs, xIdx, yIdx); 
        float3 otherPos = make_float3(otherPosWhole);
        float3 distVec = otherPos + (offset * trace) - pos;
        int otherId = *(int *) &otherPosWhole.w;

        if (myId != otherId && dot(distVec, distVec) < neighCutSqr/* && !(isExcluded(otherId, exclusions, numExclusions, maxExclusions))*/) {
   
            int xAddrNeigh = XIDX(currentNeighborIdx, sizeof(int)) * sizeof(int);
            int yIdxNeigh = YIDX(currentNeighborIdx, sizeof(int));
            if (justSorted) {
                surf2Dwrite(i, neighborlist, xAddrNeigh, yIdxNeigh);
            } else {
                int xIdxID = XIDX(otherId, sizeof(int));
                int yIdxID = YIDX(otherId, sizeof(int));
                int otherIdx = tex2D<int>(idToIdxs, xIdxID, yIdxID);
                surf2Dwrite(otherIdx, neighborlist, xAddrNeigh, yIdxNeigh);

            }
            currentNeighborIdx ++;
        }

    }
    return currentNeighborIdx;
}
__global__ void assignNeighbors(cudaTextureObject_t xs, int nAtoms, cudaTextureObject_t idToIdxs, int *nlistIdxs, int *gridCellArrayIdxs, float3 os, float3 ds, int3 ns, float3 periodic, float3 trace, float neighCutSqr, bool justSorted, cudaSurfaceObject_t neighborlist) {//, cudaTextureObject_t exclusionIdxs, cudaTextureObject_t exclusions, int maxExclusions) {
  ///  extern __shared__ int exclusions_shr[]; 
    int idx = GETIDX();
    if (idx < nAtoms) {
        int xIdxAtom = XIDX(idx, sizeof(float4));
        int yIdxAtom = YIDX(idx, sizeof(float4));
        float4 posWhole = tex2D<float4>(xs, xIdxAtom, yIdxAtom);
        int myId = *(int *)&posWhole.w;

        /*
        int exclIdxLo = tex2D<int>(exclusionIdxs, XIDX(myId, sizeof(int)), YIDX(myId, sizeof(int)));
        int exclIdxHi = tex2D<int>(exclusionIdxs, XIDX(myId+1, sizeof(int)), YIDX(myId+1, sizeof(int)));
        int numExclusions = exclIdxHi - exclIdxLo;
        for (int i=0; i<numExclusions; i++) {
            exclusions_shr[threadIdx.x*maxExclusions + i] = tex2D<int>(exclusions, XIDX(exclIdxLo + i, sizeof(int)), YIDX(exclIdxLo + i, sizeof(int)));
        }
        */



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

        int currentNeighborIdx = nlistIdxs[myIdx];
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

                                currentNeighborIdx = assignFromCell(pos, idx, myId, xs, nlistIdxs, gridCellArrayIdxs, idToIdxs, sqrIdxOtherLin, -offset, trace, neighCutSqr, currentNeighborIdx, neighborlist, justSorted);//, exclusions_shr, numExclusions, maxExclusions);

                            }
                        }
                    }
                }


            }
        }
    }
}

void GridGPU::initArrays() {
    perCellArray = GPUArrayDevice<int>(prod(ns) + 1);
    perAtomArray = GPUArrayDevice<int>(state->atoms.size()+1);
    numNeighbors = vector<int>(state->atoms.size()+1, 0);
}

GridGPU::GridGPU(State *state_, float3 ds_, float3 dsOrig_, float3 os_, int3 ns_) : state(state_), ds(ds_), dsOrig(dsOrig_), os(os_), ns(ns_), neighborlist(cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned)){
    initArrays();
};
GridGPU::GridGPU(State *state_, float dx_, float dy_, float dz_) : state(state_), neighborlist(cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned)) {
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
};



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

void GridGPU::periodicBoundaryConditions(float neighCut, bool doSort) {
    float3 ds_orig = ds;
    float3 os_orig = os;
    ds += make_float3(EPSILON, EPSILON, EPSILON); //as defined in Vector.h.  PAIN AND NUMERICAL ERROR AWAIT ALL THOSE WHO ALTER THIS LINE
    os -= make_float3(EPSILON, EPSILON, EPSILON);
    Vector nsV = Vector(make_float3(ns));
    int nAtoms = state->atoms.size();
    BoundsGPU bounds = state->boundsGPU;
    BoundsGPU boundsUnskewed = bounds.unskewed();
    float3 trace = boundsUnskewed.trace();
    int activeIdx = state->gpd.activeIdx;
    if (bounds.sides[0].y or bounds.sides[1].x) {
        Mod::unskewAtoms<<<NBLOCK(nAtoms), PERBLOCK>>>(state->gpd.xs.getSurf(activeIdx), nAtoms, bounds.sides[0], bounds.sides[1], bounds.lo);
    }
    periodicWrap<<<NBLOCK(nAtoms), PERBLOCK>>>(state->gpd.xs.getSurf(), nAtoms, boundsUnskewed);
    int numGridCells = prod(ns);
    if (numGridCells + 1 != perCellArray.n) {
        perCellArray = GPUArrayDevice<int>(numGridCells + 1);
    }
    perCellArray.memset(0);
    perAtomArray.memset(0);
    cudaDeviceSynchronize();
    countNumInGridCells<<<NBLOCK(nAtoms), PERBLOCK>>>(state->gpd.xs.getTex(), nAtoms, perCellArray.ptr, perAtomArray.ptr, os, ds, ns);
    int *gridCellCounts_h = perCellArray.get((int *) NULL);
    cudaDeviceSynchronize();

    
    cumulativeSum(gridCellCounts_h, perCellArray.n);//repurposing this as starting indexes for each grid square

    perCellArray.set(gridCellCounts_h);
    int gridIdx;
    if (doSort) {
        sortPerAtomArrays<<<NBLOCK(nAtoms), PERBLOCK>>>(

                state->gpd.xs.getSurf(activeIdx),  
                state->gpd.xs.getSurf(!activeIdx),

                state->gpd.vs(activeIdx),
                state->gpd.vs(!activeIdx),

                state->gpd.fs(activeIdx),
                state->gpd.fs(!activeIdx),

                state->gpd.fsLast(activeIdx),
                state->gpd.fsLast(!activeIdx),

                state->gpd.types.getSurf(activeIdx),
                state->gpd.types.getSurf(!activeIdx),

                state->gpd.qs.getSurf(activeIdx),
                state->gpd.qs.getSurf(!activeIdx),

                state->gpd.idToIdxs.getSurf(),

                perCellArray.ptr, perAtomArray.ptr, nAtoms, os, ds, ns
                );
        activeIdx = state->gpd.switchIdx();
        gridIdx = activeIdx;
    } else { //otherwise, just use non-active xs array as grid storage
        //gridCPU(state->gpd.xs, activeIdx, nAtoms, perCellArray, perAtomArray, os, ds, ns);
        gridNonSort<<<NBLOCK(nAtoms), PERBLOCK>>>(state->gpd.xs.getTex(activeIdx), state->gpd.xs.getSurf(!activeIdx), nAtoms, perCellArray.ptr, perAtomArray.ptr, os, ds, ns);
        gridIdx = !activeIdx;

    }

    perAtomArray.memset(0);
    countNumNeighbors<<<NBLOCK(nAtoms), PERBLOCK/*, PERBLOCK*sizeof(int)*(state->maxExclusions)*/>>>(state->gpd.xs.getTex(gridIdx), nAtoms, state->gpd.idToIdxs.getTex(), perAtomArray.ptr, perCellArray.ptr, os, ds, ns, bounds.periodic, trace, neighCut*neighCut, doSort);//, state->gpd.nlistExclusionIdxs.getTex(), state->gpd.nlistExclusions.getTex(), state->maxExclusions);
    perAtomArray.get(numNeighbors.data());
    cudaDeviceSynchronize();
    
    cumulativeSum(numNeighbors.data(), numNeighbors.size());  //okay, now this is the start index (+1 is end index) of each atom's neighbors
    perAtomArray.set(numNeighbors.data());
    int totalNumNeighbors = numNeighbors.back();
    neighborlist.resize(totalNumNeighbors); //look at method, doesn't always realloc
     
    assignNeighbors<<<NBLOCK(nAtoms), PERBLOCK/*, PERBLOCK*sizeof(int)*(state->maxExclusions)*/>>>(
            state->gpd.xs.getTex(gridIdx), 
            nAtoms, 
            state->gpd.idToIdxs.getTex(), 
            perAtomArray.ptr, 
            perCellArray.ptr, 
            os, ds, ns, bounds.periodic, trace, neighCut*neighCut, doSort, neighborlist.surf);//, state->gpd.nlistExclusionIdxs.getTex(), state->gpd.nlistExclusions.getTex(), state->maxExclusions);


    //printNeighbors<<<NBLOCK(state->atoms.size()), PERBLOCK>>>(perAtomArray.ptr, neighborlist.tex, state->atoms.size());
    /*
    int *neighCounts = perAtomArray.get((int *) NULL);
    cudaDeviceSynchronize();
   printNeighborCounts(neighCounts, state->atoms.size());
   free(neighCounts);
   */
    if (bounds.sides[0].y or bounds.sides[1].x) {
        Mod::skewAtomsFromZero<<<NBLOCK(nAtoms), PERBLOCK>>>(state->gpd.xs.getSurf(activeIdx), nAtoms, bounds.sides[0], bounds.sides[1], bounds.lo);
    }
    free(gridCellCounts_h);
    ds = ds_orig;
    os = os_orig;
 //   verifyNeighborlists(neighCut);

    


}


vector<int> toNeighborCounts(int *idxs, int nAtoms) {
    vector<int> nneigh;
    for (int i=0; i<nAtoms; i++) {
      //  cout << idxs[i] << endl;;
        nneigh.push_back(idxs[i+1]-idxs[i]);
    }
    //cout << idxs[nAtoms] << endl;
    return nneigh;
}


bool GridGPU::verifyNeighborlists(float neighCut) {
    int *nlist = neighborlist.get((int *) NULL);
    
    float cutSqr = neighCut * neighCut;
    int *neighIdxs = perAtomArray.get((int *) NULL);
    state->gpd.xs.dataToHost();
    cudaDeviceSynchronize();
    vector<int> nneigh = toNeighborCounts(neighIdxs, state->atoms.size());
    vector<float4> xs = state->gpd.xs.h_data;
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
                    atom_neighbors.push_back(*(int*)&otherWhole.w);
                }

            }
        }
        sort(atom_neighbors.begin(), atom_neighbors.end());
        cpu_neighbors.push_back(atom_neighbors);
    }
//    cout << "cpu dist is " << sqrt(lengthSqr(state->boundsGPU.minImage(xs[0]-xs[1])))  << endl;
    for (int i=0; i<xs.size(); i++) {
        vector<int> neighIds;
    //    cout << "begin end " << neighIdxs[i] << " " << neighIdxs[i+1] << endl;
        for (int nIdx=neighIdxs[i]; nIdx<neighIdxs[i+1]; nIdx++) {
      //      cout << "idx " << nlist[nIdx] << endl;
            float4 atom = xs[nlist[nIdx]];
            int id = *(int *) &atom.w;
       //     cout << "id is " << id << endl;
            neighIds.push_back(id);
        }
        sort(neighIds.begin(), neighIds.end());
        if (neighIds != cpu_neighbors[i]) {
            cout << "problem at idx " << i << " id " << *(int *) &xs[i].w << endl;
            cout << "cpu " << cpu_neighbors[i].size() << " gpu " << neighIds.size() << endl;
            for (int x : cpu_neighbors[i]) {
                cout << x << " ";
            }
            cout << endl;
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
    free(neighIdxs);
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
