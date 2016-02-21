#ifndef GPUDATA_H
#define GPUDATA_H
#include "GPUArrayPair.h"
#include "GPUArrayTexPair.h"
#include "GPUArrayDevice.h"
#define NUM_PAIR_ARRAYS 6
class GPUData {
    public:
        GPUArrayBasePair *allPairs[NUM_PAIR_ARRAYS];
        GPUArrayPair<float4> xs;
        GPUArrayPair<float4> vs;
        GPUArrayPair<float4> fs;
        GPUArrayPair<float4> fsLast;
        GPUArrayPair<uint> ids;
        GPUArrayPair<float> qs;
        GPUArrayTex<int> idToIdxs;


        GPUArray<float4> xsBuffer;
        GPUArray<float4> vsBuffer;
        GPUArray<float4> fsBuffer;
        GPUArray<float4> fsLastBuffer;
        GPUArray<uint> idsBuffer;

    //OMG REMEMBER TO ADD EACH NEW ARRAY TO THE ACTIVE DATA LIST IN INTEGRATER OR PAIN AWAITS

        GPUData() : idToIdxs(cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned)), activeIdx(0) {
           allPairs[0] = (GPUArrayBasePair *) &xs; //types (ints) are bit cast into the w value of xs.  Cast as int pls
           allPairs[1] = (GPUArrayBasePair *) &vs; //mass is stored in w value of vs.  ALWAYS do arithmatic as float3s, or you will mess up id or mass
           allPairs[2] = (GPUArrayBasePair *) &fs; //groupTags (uints) are bit cast into the w value of fs
           allPairs[3] = (GPUArrayBasePair *) &fsLast; //and one more space!
           allPairs[4] = (GPUArrayBasePair *) &ids;
           allPairs[5] = (GPUArrayBasePair *) &qs;
        }
        unsigned int activeIdx;
        unsigned int switchIdx() {
            for (int i=0; i<NUM_PAIR_ARRAYS; i++) {
                allPairs[i]->switchIdx();
            }
            activeIdx = allPairs[0]->activeIdx;
            return activeIdx;

        }
        vector<int> idToIdxsOnCopy;        

};
#endif
