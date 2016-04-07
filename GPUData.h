#pragma once
#ifndef GPUDATA_H
#define GPUDATA_H
#include "GPUArrayPair.h"
#include "GPUArrayDevice.h"
#include "Virial.h"

class GPUData {
    public:
        GPUArrayPair<float4> xs; //types (ints) are bit cast into the w value of xs.  Cast as int pls
        GPUArrayPair<float4> vs; //mass is stored in w value of vs.  ALWAYS do arithmatic as float3s, or you will mess up id or mass
        GPUArrayPair<float4> fs; //groupTags (uints) are bit cast into the w value of fs
        GPUArrayPair<float4> fsLast; //and one more space!
        GPUArrayPair<uint> ids;
        GPUArrayPair<float> qs;
        GPUArrayTex<int> idToIdxs;


        GPUArray<float4> xsBuffer;
        GPUArray<float4> vsBuffer;
        GPUArray<float4> fsBuffer;
        GPUArray<float4> fsLastBuffer;
        GPUArray<uint> idsBuffer;

        GPUArray<float> perParticleEng; //for data collection.  If we re-use per-particle arrays, we can't do async kernels to do per-group sums.  Would use less memory though
        GPUArray<Virial> perParticleVirial;

    //OMG REMEMBER TO ADD EACH NEW ARRAY TO THE ACTIVE DATA LIST IN INTEGRATER OR PAIN AWAITS

        GPUData() : idToIdxs(cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned)), activeIdx(0) {
        }
        unsigned int activeIdx;
        unsigned int switchIdx() {
            /*! \todo Find a better way to keep track of all data objects */
            xs.switchIdx();
            vs.switchIdx();
            fs.switchIdx();
            fsLast.switchIdx();
            ids.switchIdx();
            return qs.switchIdx();

        }
        vector<int> idToIdxsOnCopy;        

};
#endif
