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
#include "TypedItemHolder.h"
void export_FixPotentialMuliAtom();
//#include "FixHelpers.h"
template <class CPUVariant, class CPUMember, class CPUBase, class GPUMember, class ForcerTypeHolder, int N>
class FixPotentialMultiAtom : public Fix, public TypedItemHolder {
	public:
        FixPotentialMultiAtom (SHARED(State) state_, std::string handle_, std::string type_) : Fix(state_, handle_, "None", type_, 1), forcersGPU(1), forcerIdxs(1) {
            forceSingle = true;
            maxForcersPerBlock = 0;
        }
        //TO DO - make copies of the forcer, forcer typesbefore doing all the prepare for run modifications
        std::vector<CPUVariant> forcers;
        boost::python::list pyForcers; //to be managed by the variant-pylist interface member of parent classes
        std::unordered_map<int, ForcerTypeHolder> forcerTypes;
        GPUArrayDeviceGlobal<GPUMember> forcersGPU;
        GPUArrayDeviceGlobal<int> forcerIdxs;
        GPUArrayDeviceGlobal<ForcerTypeHolder> parameters;

        bool prepareForRun() {
            int maxExistingType = -1;
            std::unordered_map<ForcerTypeHolder, int> reverseMap;
            for (auto it=forcerTypes.begin(); it!=forcerTypes.end(); it++) {
                maxExistingType = fmax(it->first, maxExistingType);
                reverseMap[it->second] = it->first;
            }

            for (CPUVariant &forcerVar : forcers) { //collecting un-typed forcers into types
                CPUMember &forcer= boost::get<CPUMember>(forcerVar);
                if (forcer.type == -1) {
                    ForcerTypeHolder typeHolder = ForcerTypeHolder(&forcer);
                    bool parameterFound = reverseMap.find(typeHolder) != reverseMap.end();
                    if (parameterFound) {
                        forcer.type = reverseMap[typeHolder];
                    } else {
                        maxExistingType+=1;
                        forcerTypes[maxExistingType] = typeHolder;
                        reverseMap[typeHolder] = maxExistingType;
                        forcer.type = maxExistingType;

                    }
                } 
            }
            maxForcersPerBlock = copyMultiAtomToGPU<CPUVariant, CPUBase, CPUMember, GPUMember, ForcerTypeHolder, N>(state->atoms.size(), forcers, state->idxFromIdCache, &forcersGPU, &forcerIdxs, &forcerTypes, &parameters, maxExistingType);

            return true;
        }
        void setForcerType(int n, CPUMember &forcer) {
            if (n<0) {
                cout << "Tried to set bonded potential for invalid type " << n << endl;
                assert(n>=0);
            }
            ForcerTypeHolder holder (&forcer); 
            forcerTypes[n] = holder;
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
        //void createDihedral(Atom *, Atom *, Atom *, Atom *, double, double, double, double);
        //vector<pair<int, vector<int> > > neighborlistExclusions();
        //string restartChunk(string format);
        int maxForcersPerBlock;
        vector<int> getTypeIds() {
            vector<int> types;
            for (auto it = forcerTypes.begin(); it != forcerTypes.end(); it++) {
                types.push_back(it->first);
            }
            return types;

        }


};
#endif
