#ifndef FIXBONDHARMONIC_H
#define FIXBONDHARMONIC_H
#include "Bond.h"
#include "FixBond.h"
void export_FixBondHarmonic();
class FixBondHarmonic : public FixBond<BondHarmonic, BondHarmonicGPU> {
	public:
		FixBondHarmonic(SHARED(State) state_, string handle);
		//DataSet *eng;
		//DataSet *press;
        //bool prepareForRun();
       // bool dataToDevice();
     //`   bool dataToHost();
        //HEY - NEED TO IMPLEMENT REFRESHATOMS//consider that if you do so, max bonds per block could change
       // bool refreshAtoms();
        //vector<pair<int, vector<int> > > neighborlistExclusions();

        void createBond(Atom *, Atom *, float, float);
        ~FixBondHarmonic(){};
        string restartChunk(string format);
       // int maxBondsPerBlock;
        void compute(bool);
        const BondHarmonic getBond(size_t i) {
            return boost::get<BondHarmonic>(bonds[i]);
        }
        virtual vector<BondVariant> *getBonds() {
            return &bonds;
        }

};


#endif
