#pragma once
#ifndef FIXANGLEHARMONIC_H
#define FIXANGLEHARMONIC_H
#include "FixPotentialMultiAtom.h"
#include "Angle.h"
#include "VariantPyListInterface.h"

void export_FixAngleHarmonic();
class FixAngleHarmonic : public FixPotentialMultiAtom<AngleVariant, AngleHarmonic, Angle, AngleGPU, AngleHarmonicType, 3> {
	public:
        FixAngleHarmonic(SHARED(State) state_, string handle);
        VariantPyListInterface<AngleVariant, AngleHarmonic> pyListInterface;
		void compute(bool);
		//DataSet *eng;
		//DataSet *press;
        void createAngle(Atom *, Atom *, Atom *, double, double, int type_);
        void setAngleTypeCoefs(int, double, double);
        string restartChunk(string format);


};

#endif
