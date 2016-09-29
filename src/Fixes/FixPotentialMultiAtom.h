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
#include "ReadConfig.h"

#define COEF_DEFAULT INT_MAX  // invalid coef value
#include "TypedItemHolder.h"
#include "VariantPyListInterface.h"
//#include "FixHelpers.h"
template <class CPUVariant, class CPUMember, class CPUBase, class GPUMember, class ForcerTypeHolder, int N>
class FixPotentialMultiAtom : public Fix, public TypedItemHolder {
    public:
        FixPotentialMultiAtom (SHARED(State) state_, std::string handle_, std::string type_, bool forceSingle_) : Fix(state_, handle_, "None", type_, forceSingle_, false, false, 1), forcersGPU(1), forcerIdxs(1), pyListInterface(&forcers, &pyForcers)
    {
        maxForcersPerBlock = 0;
    }
        //TO DO - make copies of the forcer, forcer typesbefore doing all the prepare for run modifications
        std::vector<CPUVariant> forcers;
        boost::python::list pyForcers; //to be managed by the variant-pylist interface member of parent classes
        std::unordered_map<int, ForcerTypeHolder> forcerTypes;
        GPUArrayDeviceGlobal<GPUMember> forcersGPU;
        GPUArrayDeviceGlobal<int> forcerIdxs;
        GPUArrayDeviceGlobal<ForcerTypeHolder> parameters;
        VariantPyListInterface<CPUVariant, CPUMember> pyListInterface;
        int sharedMemSizeForParams;
        bool usingSharedMemForParams;
        int maxForcersPerBlock;
        virtual bool prepareForRun() {
            int maxExistingType = -1;
            std::unordered_map<ForcerTypeHolder, int> reverseMap;
            for (auto it=forcerTypes.begin(); it!=forcerTypes.end(); it++) {
                maxExistingType = std::fmax(it->first, maxExistingType);
                reverseMap[it->second] = it->first;
            }

            for (CPUVariant &forcerVar : forcers) { //collecting un-typed forcers into types
                CPUMember &forcer= boost::get<CPUMember>(forcerVar);
                if (forcer.type == -1) {
                    //cout << "gotta do" << endl;
                    //cout << "max existing type " << maxExistingType  << endl;
                    //
                    //to do: make it so I just cast forcer as a type.  Gave nans last time I tried it
                    ForcerTypeHolder typeHolder = ForcerTypeHolder(&forcer);
                    std::cout << typeHolder.getInfoString() << std::endl;
                    bool parameterFound = reverseMap.find(typeHolder) != reverseMap.end();
                    //cout << "is found " << parameterFound << endl;
                    if (parameterFound) {
                        forcer.type = reverseMap[typeHolder];
                    } else {
                        maxExistingType+=1;
                        forcerTypes[maxExistingType] = typeHolder;
                        reverseMap[typeHolder] = maxExistingType;
                        forcer.type = maxExistingType;
                        //cout << "assigning type of " << forcer.type << endl;

                    }
                } 
            }
            maxForcersPerBlock = copyMultiAtomToGPU<CPUVariant, CPUBase, CPUMember, GPUMember, ForcerTypeHolder, N>(state->atoms.size(), forcers, state->idToIdx, &forcersGPU, &forcerIdxs, &forcerTypes, &parameters, maxExistingType);


            setSharedMemForParams(); 
            return true;
        } 
        void setForcerType(int n, CPUMember &forcer) {
            if (n < 0) {
                std::cout << "Tried to set bonded potential for invalid type " << n << std::endl;
                assert(n >= 0);
            }
            ForcerTypeHolder holder (&forcer); 
            forcerTypes[n] = holder;
        }

	std::string restartChunk(std::string format) {
	  std::stringstream ss;
	  ss << "<types>\n";
	  for (auto it = forcerTypes.begin(); it != forcerTypes.end(); it++) {
	    ss << "<" << "type id='" << it->first << "'";
	    ss << forcerTypes[it->first].getInfoString() << "'/>\n";
	  }
	  ss << "</types>\n";
	  ss << "<members>\n";
	  for (CPUVariant &forcerVar : forcers) {
	    CPUMember &forcer= boost::get<CPUMember>(forcerVar);
	    ss << forcer.getInfoString();
	  }
	  ss << "</members>\n";
	  return ss.str();
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

        std::vector<int> getTypeIds() {
            std::vector<int> types;
            for (auto it = forcerTypes.begin(); it != forcerTypes.end(); it++) {
                types.push_back(it->first);
            }
            return types;
        }
        void duplicateMolecule(std::map<int, int> &oldToNew) {
            int ii = forcers.size();
            for (int i=0; i<ii; i++) {
                CPUMember &forcer = boost::get<CPUMember>(forcers[i]);
                std::array<int, N> &ids = forcer.ids;
                std::array<int, N> idsNew = ids;
                for (int j=0; j<N; j++) {
                    if (oldToNew.find(ids[j]) != oldToNew.end()) {
                        idsNew[j] = oldToNew[ids[j]];
                    }
                }
                if (ids != idsNew) {
                    CPUMember duplicate = forcer;
                    duplicate.ids = idsNew;
                    forcers.push_back(duplicate);
                    pyListInterface.updateAppendedMember(false);


                }
            }
            pyListInterface.requestRefreshPyList();            
            
        }

        void setSharedMemForParams() {
            int size = parameters.size() * sizeof(ForcerTypeHolder);
            //<= 3 is b/c of threshold for using redundant calcs
            if (size + int(N<=3) * maxForcersPerBlock*sizeof(GPUMember)> state->devManager.prop.sharedMemPerBlock) {
                usingSharedMemForParams = false;
                sharedMemSizeForParams = 0;
            } else {
                usingSharedMemForParams = true;
                sharedMemSizeForParams = size;
            }

        }
        void deleteAtom(Atom *a) {
            int deleteId = a->id;
            for (int i=forcers.size()-1; i>=0; i--) {
                CPUMember &forcer= boost::get<CPUMember>(forcers[i]);
                bool deleteForcer = false;
                for (int id : forcer.ids) {
                    if (id == deleteId) {
                        deleteForcer = true;
                        break;
                    }
                }
                if (deleteForcer) {
                    forcers.erase(forcers.begin()+i, forcers.begin()+i+1);
                    pyListInterface.removeMember(i);
                    pyListInterface.requestRefreshPyList();
                }
            }
        }
};

#endif
