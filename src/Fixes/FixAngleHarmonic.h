#pragma once
#ifndef FIXANGLEHARMONIC_H
#define FIXANGLEHARMONIC_H

#include "FixPotentialMultiAtom.h"
#include "Angle.h"
#include "AngleEvaluatorHarmonic.h"

void export_FixAngleHarmonic();

class FixAngleHarmonic : public FixPotentialMultiAtom<AngleVariant, AngleHarmonic, Angle, AngleGPU, AngleHarmonicType, 3> {

private:
    AngleEvaluatorHarmonic evaluator; 
public:

    FixAngleHarmonic(boost::shared_ptr<State> state_, std::string handle);

    virtual void compute(int) override;
    virtual void singlePointEng(real *) override;

    void createAngle(Atom *, Atom *, Atom *, double, double, int type_);
    void setAngleTypeCoefs(int, double, double);

    virtual bool readFromRestart() override;

};

#endif
