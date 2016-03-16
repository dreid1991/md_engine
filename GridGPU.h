#ifndef GRID_GPU
#define GRID_GPU


#include "cutils_math.h"
#include "GPUArray.h"
#include "GPUArrayTexDevice.h"
#include <unordered_map>
#include <unordered_set>
#include <set>

#define EXCL_MASK (~(3<<30));
//okay, this is going to contain all the kernels needed to do gridding
//can also have it contain the 3d grid for neighbor int2 s
class State;
class GridGPU {
    bool is2d;
    void initArrays();
    void initStream();
    bool verifyNeighborlists(float neighCut);
    bool streamCreated;
    bool checkSorting(int gridIdx, int *gridIdxs, GPUArrayDevice<int> &grid);
    GPUArrayDevice<float4> xsLastBuild;
    GPUArray<int> buildFlag;
    public: 
        GPUArray<int> perCellArray;
        GPUArray<int> perBlockArray;
        GPUArray<int> perAtomArray; //during runtime this is the starting (+1 is ending) index for each neighbor
        float3 ds;
        float3 dsOrig;
        float3 os;
        int3 ns;
        GPUArrayDevice<uint> neighborlist;
        State *state; 
        GridGPU(State *state_, float dx, float dy, float dz);
        GridGPU(State *state_, float3 ds_, float3 dsOrig_, float3 os_, int3 ns_);
        GridGPU(); // NEED TO CREATE STREAM OR WILL BE TRYING TO DESTROY NONEXISTANT STREAM
        ~GridGPU();
        //need set2d function
        void handleExclusions();
        void periodicBoundaryConditions(float neighCut, bool doSort, bool forceBuild=false);
        //exclusion list stuff     
		typedef map<int, vector<set<int>>> ExclusionList; //is ordered to make looping over by id in order easier
		bool closerThan(const ExclusionList &exclude,
						int atomid, int otherid, int16_t depthi);
		ExclusionList generateExclusionList(const int16_t maxDepth);
      //  ExclusionList exclusionList;
        GPUArrayDevice<int> exclusionIndexes;
        GPUArrayDevice<uint> exclusionIds;
        int maxExclusionsPerAtom;
        int numChecksSinceLastBuild;
        cudaStream_t rebuildCheckStream;
        void copyPositionsAsync();
};
#endif
