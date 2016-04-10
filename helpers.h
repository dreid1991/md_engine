#pragma once
#ifndef HELPERS_H
#define HELPERS_H
#include "GPUArrayDeviceGlobal.h"
#include <vector>
#include "Atom.h"
#include <boost/variant.hpp>
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
int copyMultiAtomToGPU(vector<Atom> &atoms, vector<SRCVar> &src, GPUArrayDeviceGlobal<DEST> *dest, GPUArrayDeviceGlobal<int> *destIdxs) {
    vector<int> idxs(atoms.size()+1, 0); //started out being used as counts
    vector<int> numAddedPerAtom(atoms.size(), 0);
    //so I can arbitrarily order.  I choose to do it by the the way atoms happen to be sorted currently.  Could be improved.
    for (SRCVar &sVar : src) {
        SRCFull &s = boost::get<SRCFull>(sVar);
        for (int i=0; i<N; i++) {
            idxs[s.atoms[i] - atoms.data()]++;
        }
        
    }
    cumulativeSum(idxs.data(), atoms.size()+1);  
    vector<DEST> destHost(idxs.back());
    for (SRCVar &sVar : src) {
        SRCFull &s = boost::get<SRCFull>(sVar);
        int atomIds[N];
        int atomIndexes[N];
        for (int i=0; i<N; i++) {
            atomIds[i] = s.atoms[i]->id;
            atomIndexes[i] = s.atoms[i] - atoms.data();

        }
        DEST d;
        d.takeIds(atomIds);
        d.takeValues(s);
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
    for (int i=0; i<atoms.size(); i+=PERBLOCK) {
        maxPerBlock = fmax(maxPerBlock, idxs[fmin(i+PERBLOCK+1, idxs.size()-1)] - idxs[i]);
    }
    return maxPerBlock;

}
#endif
