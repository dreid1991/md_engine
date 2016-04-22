#pragma once 
#ifndef FIXBOND_H
#define FIXBOND_H

#include "Fix.h"
#include "Bond.h"

#include "helpers.h" //cumulative sum
#include <unordered_map>
#include "TypedItemHolder.h"

template <class SRC, class DEST>
int copyBondsToGPU(vector<Atom> &atoms, vector<BondVariant> &src, GPUArrayDeviceGlobal<DEST> *dest, GPUArrayDeviceGlobal<int> *destIdxs) {
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




template <class CPUMember, class GPUMember>
class FixBond : public Fix, public TypedItemHolder {
    public:
        GPUArrayDeviceGlobal<GPUMember> bondsGPU;
        GPUArrayDeviceGlobal<int> bondIdxs;
        vector<BondVariant> bonds;
        boost::python::list pyBonds;
        int maxBondsPerBlock;
        std::unordered_map<int, CPUMember> forcerTypes;
        FixBond(SHARED(State) state_, string handle_, string groupHandle_, string type_, int applyEvery_) : Fix(state_, handle_, groupHandle_, type_, applyEvery_) {
            forceSingle = true;
            maxBondsPerBlock = 0;
        }
        void setForcerType(int n, CPUMember &forcer) {
            if (n<0) {
                cout << "Tried to set bonded potential for invalid type " << n << endl;
                assert(n>=0);
            }
            forcerTypes[n] = forcer;
        }

        bool refreshAtoms() {
            vector<int> idxFromIdCache = state->idxFromIdCache;
            vector<Atom> &atoms = state->atoms;
            for (BondVariant &bv : bonds) {
                CPUMember *cpuMem = &get<CPUMember>(bv);
                cpuMem->atoms[0] = &atoms[idxFromIdCache[cpuMem->id1]];
                cpuMem->atoms[1] = &atoms[idxFromIdCache[cpuMem->id2]];
            }
            return true;
        }

        bool prepareForRun() {
            vector<Atom> &atoms = state->atoms;
            refreshAtoms();
            for (BondVariant &bondVar: bonds) { //applying types to individual elements
                CPUMember &bond = boost::get<CPUMember>(bondVar);
                if (bond.type != -1) {
                    auto it = forcerTypes.find(bond.type);
                    if (it == forcerTypes.end()) {
                        cout << "Invalid bonded potential type " << bond.type << endl;
                        assert(it != forcerTypes.end());
                    }
                    bond.takeValues(it->second); 
                }
            }
            maxBondsPerBlock = copyBondsToGPU<CPUMember, GPUMember>(atoms, bonds, &bondsGPU, &bondIdxs);

            return true;

        }
        vector<int> getTypeIds() {
            vector<int> ids;
            for (auto it=forcerTypes.begin(); it!=forcerTypes.end(); it++) {
                ids.push_back(it->first);
            }
            return ids;
        }
        


};

#endif
