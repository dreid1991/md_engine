#include "Improper.h"

ImproperHarmonic::ImproperHarmonic(Atom *a, Atom *b, Atom *c, Atom *d, double k_, double thetaEq_) {
    atoms[0] = a;
    atoms[1] = b;
    atoms[2] = c;
    atoms[4] = d;
    k = k_;
    thetaEq = thetaEq_;

}

void ImproperHarmonicGPU::takeIds(int *ids_) {
    ids[0] = ids_[0];
    ids[1] = ids_[1];
    ids[2] = ids_[2];
    ids[3] = ids_[3];
}
void ImproperHarmonicGPU::takeValues(ImproperHarmonic &other) {
    k = other.k;
    thetaEq = other.thetaEq;
}
