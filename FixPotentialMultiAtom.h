#pragma once
#ifndef FIXPOTENTIALMULTIATOM_H
#define FIXPOTENTIALMULTIATOM_H
#include <array>
#include <vector>
#include "GPUArrayDeviceGlobal.h"
#include "State.h"
#include "Fix.h"
#include <unordered_map>
#include "helpers.h"
#include <boost/variant.hpp>
#include <climits>
#define COEF_DEFAULT INT_MAX //invaled coef value
//#include "FixHelpers.h"
//FixDihedralOPLS::FixDihedralOPLS(SHARED(State) state_, string handle) : Fix(state_, handle, string("None"), dihedralOPLSType, 1), dihedralsGPU(1), dihedralIdxs(1)  {
template <class CPUVariant, class CPUMember, class GPUMember, int N>
class FixPotentialMultiAtom : public Fix {
	public:
        FixPotentialMultiAtom (SHARED(State) state_, std::string handle_, std::string type_) : Fix(state_, handle_, "None", type_, 1), forcersGPU(1), forcerIdxs(1) {
            forceSingle = true;
            maxForcersPerBlock = 0;
        }
        std::vector<std::array<int, N> > forcerAtomIds;
        std::vector<CPUVariant> forcers;
        std::unordered_map<int, CPUMember> forcerTypes;
        GPUArrayDeviceGlobal<GPUMember> forcersGPU;
        GPUArrayDeviceGlobal<int> forcerIdxs;
		//DataSet *eng;
        //DataSet *press;
        bool prepareForRun() {
            std::vector<Atom> &atoms = state->atoms;
            refreshAtoms();
            for (CPUVariant &forcerVar : forcers) { //applying types to individual elements
                CPUMember &forcer= boost::get<CPUMember>(forcerVar);
                if (forcer.type != -1) {
                    auto it = forcerTypes.find(forcer.type);
                    if (it == forcerTypes.end()) {
                        cout << "Invalid bonded potential type " << forcer.type << endl;
                        assert(it != forcerTypes.end());
                    }
                    forcer.takeValues(it->second); 
                }
            }
            maxForcersPerBlock = copyMultiAtomToGPU<CPUVariant, CPUMember, GPUMember, N>(atoms, forcers, &forcersGPU, &forcerIdxs);

            return true;
        }
        void setForcerType(int n, CPUMember &forcer) {
            if (n<0) {
                cout << "Tried to set bonded potential for invalid type " << n << endl;
                assert(n>=0);
            }
            forcerTypes[n] = forcer;
        }

        void atomsValid(std::vector<Atom *> &atoms) {
            for (int i=0; i<atoms.size(); i++) {
                if (!state->validAtom(atoms[i])) {
                    cout << "Tried to create for " << handle << " but atom " << i << " was invalid" << endl;
                    assert(false);
                }
            }
        }
        bool downloadFromRun(){return true;};
        //HEY - NEED TO IMPLEMENT REFRESHATOMS
        bool refreshAtoms() {
            std::vector<int> idxFromIdCache = state->idxFromIdCache;
            std::vector<Atom> &atoms = state->atoms;
            for (int i=0; i<forcerAtomIds.size(); i++) {
                CPUMember &forcer = boost::get<CPUMember>(forcers[i]);
                std::array<int, N> &ids = forcerAtomIds[i];
                for (int j=0; j<N; j++) {
                    forcer.atoms[j] = &atoms[idxFromIdCache[ids[j]]];
                }
            }
            return forcerAtomIds.size() == forcers.size();

        }
        //void createDihedral(Atom *, Atom *, Atom *, Atom *, double, double, double, double);
        //vector<pair<int, vector<int> > > neighborlistExclusions();
        //string restartChunk(string format);
        int maxForcersPerBlock;


};
#endif
