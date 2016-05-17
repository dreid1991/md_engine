#pragma once
#ifndef FIXPOTENTIALMULTIATOM_H
#define FIXPOTENTIALMULTIATOM_H

#include <array>
#include <vector>
#include <climits>
#include <unordered_map>

#include <boost/variant.hpp>

#include "GPUArrayDeviceGlobal.h"
#include "State.h"
#include "Fix.h"
#include "helpers.h"

#define COEF_DEFAULT INT_MAX  // invalid coef value
#include "TypedItemHolder.h"
//#include "FixHelpers.h"

void export_FixPotentialMuliAtom();

template <class CPUVariant, class CPUMember, class GPUMember, int N>
class FixPotentialMultiAtom : public Fix, public TypedItemHolder {

public:
    std::vector<CPUVariant> forcers;
    boost::python::list pyForcers;  // to be managed by the variant-pylist interface member of parent classes
    std::unordered_map<int, CPUMember> forcerTypes;
    GPUArrayDeviceGlobal<GPUMember> forcersGPU;
    GPUArrayDeviceGlobal<int> forcerIdxs;
    int maxForcersPerBlock;
    //DataSet *eng;
    //DataSet *press;

    FixPotentialMultiAtom(boost::shared_ptr<State> state_,
                          std::string handle_, std::string type_, bool forceSingle_)
      : Fix(state_, handle_, "None", type_, forceSingle_, 1),
        forcersGPU(1), forcerIdxs(1)
    {
        maxForcersPerBlock = 0;
    }

    bool prepareForRun() {
        for (CPUVariant &forcerVar : forcers) {  // applying types to individual elements
            CPUMember &forcer = boost::get<CPUMember>(forcerVar);
            if (forcer.type != -1) {
                auto it = forcerTypes.find(forcer.type);
                if (it == forcerTypes.end()) {
                    std::cout << "Invalid bonded potential type " << forcer.type << std::endl;
                    assert(it != forcerTypes.end());
                }
                forcer.takeParameters(it->second);
            }
        }
        maxForcersPerBlock = copyMultiAtomToGPU<CPUVariant, CPUMember, GPUMember, N>(
                                state->atoms.size(), forcers, state->idxFromIdCache,
                                &forcersGPU, &forcerIdxs);

        return true;
    }

    void setForcerType(int n, CPUMember &forcer) {
        if (n < 0) {
            std::cout << "Tried to set bonded potential for invalid type " << n << std::endl;
            assert(n >= 0);
        }
        forcerTypes[n] = forcer;
    }

    void atomsValid(std::vector<Atom *> &atoms) {
        for (int i=0; i<atoms.size(); i++) {
            if (!state->validAtom(atoms[i])) {
                std::cout << "Tried to create for " << handle
                          << " but atom " << i << " was invalid" << std::endl;
                assert(false);
            }
        }
    }

    //HEY - NEED TO IMPLEMENT REFRESHATOMS
    //void createDihedral(Atom *, Atom *, Atom *, Atom *, double, double, double, double);
    //std::vector<pair<int, std::vector<int> > > neighborlistExclusions();
    //std::string restartChunk(std::string format);
    std::vector<int> getTypeIds() {
        std::vector<int> types;
        for (auto it = forcerTypes.begin(); it != forcerTypes.end(); it++) {
            types.push_back(it->first);
        }
        return types;
    }

};

#endif
