#pragma once
#include "FixPotentialMultiAtom.h"
#include "Improper.h"

void export_FixImproperHarmonic();
class FixImproperHarmonic: public FixPotentialMultiAtom<ImproperHarmonic, ImproperHarmonicGPU, 4> {
	public:
        FixImproperHarmonic(SHARED(State) state_, string handle);
		void compute(bool);
		//DataSet *eng;
		//DataSet *press;
        void createImproper(Atom *, Atom *, Atom *, Atom *, double, double, int);
        //vector<pair<int, vector<int> > > neighborlistExclusions();
        void setImproperTypeCoefs(int, double, double);
        string restartChunk(string format);


};

