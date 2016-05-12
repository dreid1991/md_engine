#pragma once
#ifndef FIXIMPROPERHARMONIC_H
#define FIXIMPROPERHARMONIC_H

#include "FixPotentialMultiAtom.h"
#include "Improper.h"
#include "VariantPyListInterface.h"

void export_FixImproperHarmonic();
class FixImproperHarmonic: public FixPotentialMultiAtom<ImproperVariant, ImproperHarmonic, Improper, ImproperGPU, ImproperHarmonicType, 4> {
	public:
        FixImproperHarmonic(SHARED(State) state_, string handle);
        VariantPyListInterface<ImproperVariant, ImproperHarmonic> pyListInterface;
		void compute(bool);
		//DataSet *eng;
		//DataSet *press;
        void createImproper(Atom *, Atom *, Atom *, Atom *, double, double, int);
        //vector<pair<int, vector<int> > > neighborlistExclusions();
        void setImproperTypeCoefs(int, double, double);
        string restartChunk(string format);


};

#endif
