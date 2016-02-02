#include "Angle.h"

AngleHarmonic::AngleHarmonic(Atom *a, Atom *b, Atom *c, double k_, double thetaEq_) {
    atoms[0] = a;
    atoms[1] = b;
    atoms[2] = c;
    k = k_;
    thetaEq = thetaEq_;

}

void AngleHarmonicGPU::takeIds(int *ids_) {
    ids[0] = ids_[0];
    ids[1] = ids_[1];
    ids[2] = ids_[2];
}
void AngleHarmonicGPU::takeValues(AngleHarmonic &other) {
    k = other.k;
    thetaEq = other.thetaEq;
}
