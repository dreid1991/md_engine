#pragma once 
#include "Fix.h"
#include "Bond.h"

#include "helpers.h" //cumulative sum

template <class SRC, class DEST>
int copyBondsToGPU(vector<Atom> &atoms, vector<BondVariant> &src, GPUArrayDevice<DEST> *dest, GPUArrayDevice<int> *destIdxs) {
    vector<int> idxs(atoms.size()+1, 0); //started out being used as counts
    vector<int> numAddedPerAtom(atoms.size(), 0);
    //so I can arbitrarily order.  I choose to do it by the the way atoms happen to be sorted currently.  Could be improved.
    for (BondVariant &s : src) {
        idxs[get<SRC>(s).atoms[0] - atoms.data()]++;
        idxs[get<SRC>(s).atoms[1] - atoms.data()]++;
    }
    cumulativeSum(idxs.data(), atoms.size()+1);  
    vector<DEST> destHost(idxs.back());
    for (BondVariant &sv : src) {
        SRC &s = get<SRC>(sv);
        int bondAtomIds[2];
        int bondAtomIndexes[2];
        bondAtomIds[0] = s.atoms[0]->id;
        bondAtomIds[1] = s.atoms[1]->id;
        bondAtomIndexes[0] = s.atoms[0] - atoms.data();
        bondAtomIndexes[1] = s.atoms[1] - atoms.data();
        for (int i=0; i<2; i++) {
            DEST a;
            a.myId = bondAtomIds[i];
            a.idOther = bondAtomIds[!i];
            a.takeValues(s);
            destHost[idxs[bondAtomIndexes[i]] + numAddedPerAtom[bondAtomIndexes[i]]] = a;
            numAddedPerAtom[bondAtomIndexes[i]]++;
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

template <class CPUType, class GPUType>
class FixBond : public Fix {
    public:
        vector<int2> bondAtomIds;
        GPUArrayDevice<GPUType> bondsGPU;
        GPUArrayDevice<int> bondIdxs;
        vector<BondVariant> bonds;
        int maxBondsPerBlock;
        FixBond(SHARED(State) state_, string handle_, string groupHandle_, string type_, int applyEvery_) 
          : Fix(state_, handle_, groupHandle_, type_, applyEvery_) {
            forceSingle = true;
            maxBondsPerBlock = 0;
        }
        bool refreshAtoms() {
            vector<int> idxFromIdCache = state->idxFromIdCache;
            vector<Atom> &atoms = state->atoms;
            for (int i=0; i<bondAtomIds.size(); i++) {
                int2 ids = bondAtomIds[i];
                get<CPUType>(bonds[i]).atoms[0] = &atoms[idxFromIdCache[ids.x]];//state->atomFromId(ids.x);
                get<CPUType>(bonds[i]).atoms[1] = &atoms[idxFromIdCache[ids.y]];//state->atomFromId(ids.y);
            }
            return bondAtomIds.size() == bonds.size();
        }

        bool prepareForRun() {
            vector<Atom> &atoms = state->atoms;
            refreshAtoms();
            maxBondsPerBlock = copyBondsToGPU<CPUType, GPUType>(atoms, bonds, &bondsGPU, &bondIdxs);

            return true;

        }

};

