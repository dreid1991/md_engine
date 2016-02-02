#ifndef FIXPOTENTIALMULTIATOM_H
#define FIXPOTENTIALMULTIATOM_H
#include <array>
#include <vector>
#include "GPUArrayDevice.h"
#include "State.h"
#include "FixHelpers.h"
using namespace std;
//FixDihedralOPLS::FixDihedralOPLS(SHARED(State) state_, string handle) : Fix(state_, handle, string("None"), dihedralOPLSType, 1), dihedralsGPU(1), dihedralIdxs(1)  {
template <class CPUMember, class GPUMember, int N>
class FixPotentialMultiAtom : public Fix {
	public:
        FixPotentialMultiAtom (SHARED(State) state_, string handle_, string type_) : Fix(state_, handle_, "None", type_, 1), forcersGPU(1), forcerIdxs(1) {
            forceSingle = true;
            maxForcersPerBlock = 0;
        }
        vector<std::array<int, N> > forcerAtomIds;
        vector<CPUMember> forcers;
        GPUArrayDevice<GPUMember> forcersGPU;
        GPUArrayDevice<int> forcerIdxs;
		//DataSet *eng;
        //DataSet *press;
        bool prepareForRun() {
            vector<Atom> &atoms = state->atoms;
            refreshAtoms();
            maxForcersPerBlock = copyMultiAtomToGPU<CPUMember, GPUMember, N>(atoms, forcers, &forcersGPU, &forcerIdxs);

            return true;
        }

        void atomsValid(vector<Atom *> &atoms) {
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
            vector<int> idxFromIdCache = state->idxFromIdCache;
            vector<Atom> &atoms = state->atoms;
            for (int i=0; i<forcerAtomIds.size(); i++) {
                std::array<int, N> &ids = forcerAtomIds[i];
                for (int j=0; j<N; j++) {
                    forcers[i].atoms[j] = &atoms[idxFromIdCache[ids[j]]];
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
