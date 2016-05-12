#pragma once
#ifndef HELPERS_H
#define HELPERS_H
#include "GPUArrayDeviceGlobal.h"
#include <vector>
#include "Atom.h"
#include <boost/variant.hpp>
#include <unordered_map>
#include <array>
using namespace std;
template <class T, class K>
void cumulativeSum(T *data, K n) {
    int currentVal= 0;
    for (int i=0; i<n-1; i++) { 
        int numInCell = data[i];
        data[i] = currentVal;
        currentVal += numInCell;
    }
    data[n-1] = currentVal; //okay, so now nth place has grid's starting Idx, n+1th place has ending
}

template <class SRCVar, class SRCBase, class SRCFull, class DEST, class TYPEHOLDER, int N>
int copyMultiAtomToGPU(int nAtoms, vector<SRCVar> &src, vector<int> &idxFromIdCache, GPUArrayDeviceGlobal<DEST> *dest, GPUArrayDeviceGlobal<int> *destIdxs, std::unordered_map<int, TYPEHOLDER> *forcerTypes, GPUArrayDeviceGlobal<TYPEHOLDER> *parameters, int maxExistingType) {
    vector<int> idxs(nAtoms+1, 0); //started out being used as counts
    vector<int> numAddedPerAtom(nAtoms, 0);
    //so I can arbitrarily order.  I choose to do it by the the way atoms happen to be sorted currently.  Could be improved.
    for (SRCVar &sVar : src) {
        SRCFull &s = boost::get<SRCFull>(sVar);
        for (int i=0; i<N; i++) {
            int id = s.ids[i];
            idxs[idxFromIdCache[id]]++;
        }
        
    }
    cumulativeSum(idxs.data(), nAtoms+1);  
    vector<DEST> destHost(idxs.back());
    for (SRCVar &sVar : src) {
        SRCFull &s = boost::get<SRCFull>(sVar);
        SRCBase *base = (SRCBase *) &s;
        std::array<int, N> atomIds = s.ids;
        std::array<int, N> atomIndexes;
        for (int i=0; i<N; i++) {
            atomIndexes[i] = idxFromIdCache[atomIds[i]];
        }
        DEST d;
        d.takeIds(base);
        uint32_t type = s.type;
        assert(N <= 8); //three bits for which idx, allowing up to 8 member forcers, can be changed later
        for (int i=0; i<N; i++) {
            DEST dForIth = d;
            dForIth.type = type | (i << 29);
            destHost[idxs[atomIndexes[i]] + numAddedPerAtom[atomIndexes[i]]] = dForIth;
            numAddedPerAtom[atomIndexes[i]]++;
        }
    }
    *dest = GPUArrayDeviceGlobal<DEST>(destHost.size());
    dest->set(destHost.data());
    *destIdxs = GPUArrayDeviceGlobal<int>(idxs.size());
    destIdxs->set(idxs.data());

    //getting max # bonds per block
    int maxPerBlock = 0;
    for (int i=0; i<nAtoms; i+=PERBLOCK) {
        maxPerBlock = fmax(maxPerBlock, idxs[fmin(i+PERBLOCK+1, idxs.size()-1)] - idxs[i]);
    }


    //now copy types
    //if user is silly and specifies huge types values, these kernels could crash
    //should add error messages and such about that
    vector<TYPEHOLDER> types(maxExistingType+1);
    for (auto it = forcerTypes->begin(); it!= forcerTypes->end(); it++) {
        types[it->first] = it->second;
    }
    *parameters = GPUArrayDeviceGlobal<TYPEHOLDER>(types.size());
    parameters->set(types.data());
    cudaDeviceSynchronize();


    return maxPerBlock;

}
#endif
