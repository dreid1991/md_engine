#ifndef GRID_GPU
#define GRID_GPU


#include "Mod.h"
#include "BoundsGPU.h"
#include "cutils_math.h"
#include "GPUArrayDevice.h"
#include "GPUArrayTexDevice.h"
//okay, this is going to contain all the kernels needed to do gridding
//can also have it contain the 3d grid for neighbor int2 s
class State;
class GridGPU {
    bool is2d;
    void initArrays();
    bool verifyNeighborlists(float neighCut);

    vector<int> numNeighbors;
    bool checkSorting(int gridIdx, int *gridIdxs, GPUArrayDevice<int> &grid);
    public: 
        GPUArrayDevice<int> perCellArray;
        float3 ds;
        float3 dsOrig;
        float3 os;
        int3 ns;
        GPUArrayTexDevice<int> neighborlist;
        GPUArrayDevice<int> perAtomArray; //during runtime this is the starting (+1 is ending) index for each neighbor
        State *state; 
        GridGPU(State *state_, float dx, float dy, float dz);
        GridGPU(State *state_, float3 ds_, float3 dsOrig_, float3 os_, int3 ns_);
        GridGPU() {};
        //need set2d function
        void periodicBoundaryConditions(float neighCut, bool doSort);
};
#endif
