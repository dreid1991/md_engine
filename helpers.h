#pragma once
#ifndef HELPERS_H
#define HELPERS_H
#include "GPUArrayDevice.h"
#include <vector>
#include "Atom.h"
#include <boost/variant.hpp>
using namespace std;
void cumulativeSum(int *data, int n);

template <class SRCVar, class SRCFull, class DEST, int N>
int copyMultiAtomToGPU(vector<Atom> &atoms, vector<SRCVar> &src, GPUArrayDevice<DEST> *dest, GPUArrayDevice<int> *destIdxs) {
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
    *dest = GPUArrayDevice<DEST>(destHost.size());
    dest->set(destHost.data());
    *destIdxs = GPUArrayDevice<int>(idxs.size());
    destIdxs->set(idxs.data());

    //getting max # bonds per block
    int maxPerBlock = 0;
    for (int i=0; i<atoms.size(); i+=PERBLOCK) {
        maxPerBlock = fmax(maxPerBlock, idxs[fmin(i+PERBLOCK+1, idxs.size()-1)] - idxs[i]);
    }
    return maxPerBlock;

}
#endif
