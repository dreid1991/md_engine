#pragma once
#ifndef FIXANGLEHARMONIC_H
#define FIXANGLEHARMONIC_H

#include "FixPotentialMultiAtom.h"
#include "Angle.h"
#include "VariantPyListInterface.h"

void export_FixAngleHarmonic();

class FixAngleHarmonic : public FixPotentialMultiAtom<AngleVariant, AngleHarmonic, Angle, AngleGPU, AngleHarmonicType, 3> {

public:
    //DataSet *eng;
    //DataSet *press;
    VariantPyListInterface<AngleVariant, AngleHarmonic> pyListInterface;

    FixAngleHarmonic(boost::shared_ptr<State> state_, std::string handle);

    void compute(bool);

    void createAngle(Atom *, Atom *, Atom *, double, double, int type_);
    void setAngleTypeCoefs(int, double, double);

    std::string restartChunk(std::string format);

};

#endif
