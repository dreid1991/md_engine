#pragma once
#ifndef HELPERS_H
#define HELPERS_H
#include "GPUArrayDeviceGlobal.h"
#include <vector>
#include "Atom.h"
#include <boost/variant.hpp>
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

template <class SRCVar, class SRCFull, class DEST, int N>
int copyMultiAtomToGPU(int nAtoms, vector<SRCVar> &src, vector<int> &idxFromIdCache, GPUArrayDeviceGlobal<DEST> *dest, GPUArrayDeviceGlobal<int> *destIdxs) {
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
        std::array<int, N> atomIds = s.ids;
        std::array<int, N> atomIndexes;
        for (int i=0; i<N; i++) {
            atomIndexes[i] = idxFromIdCache[atomIds[i]];
        }
        DEST d;
        d.takeIds(s);
        d.takeParameters(s);
        for (int i=0; i<N; i++) {
            DEST dForIth = d;
            dForIth.myIdx = i;
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
    return maxPerBlock;

}
#endif
