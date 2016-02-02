#ifndef FIXBONDHARMONIC_H
#define FIXBONDHARMONIC_H
#include "Fix.h"
#include "Bond.h"

void export_FixBondHarmonic();
class FixBondHarmonic : public Fix {
	public:
		FixBondHarmonic(SHARED(State) state_, string handle);
        vector<int2> bondAtomIds;
        GPUArrayDevice<BondHarmonicGPU> bondsGPU;
        GPUArrayDevice<int> bondIdxs;
		void compute();
		//DataSet *eng;
		//DataSet *press;
        bool prepareForRun();
        bool downloadFromRun(){return true;};
       // bool dataToDevice();
     //`   bool dataToHost();
        //HEY - NEED TO IMPLEMENT REFRESHATOMS//consider that if you do so, max bonds per block could change
        bool refreshAtoms();
        //vector<pair<int, vector<int> > > neighborlistExclusions();

        void createBond(Atom *, Atom *, float, float);
        ~FixBondHarmonic(){};
        string restartChunk(string format);
        int maxBondsPerBlock;
        vector<BondVariant> bonds;
        const BondHarmonic getBond(size_t i) {
            return boost::get<BondHarmonic>(bonds[i]);
        }
        virtual vector<BondVariant> *getBonds() {
            return &bonds;
        }

};


#endif
