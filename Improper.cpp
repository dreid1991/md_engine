#include "Improper.h"

ImproperHarmonic::ImproperHarmonic(Atom *a, Atom *b, Atom *c, Atom *d, double k_, double thetaEq_, int type_) {
    atoms[0] = a;
    atoms[1] = b;
    atoms[2] = c;
    atoms[4] = d;
    k = k_;
    thetaEq = thetaEq_;
    type = type_;

}
ImproperHarmonic::ImproperHarmonic(double k_, double thetaEq_, int type_) {
    for (int i=0; i<4; i++) {
        atoms[i] = (Atom *) NULL;
    }
    k = k_;
    thetaEq = thetaEq_;
    type = type_;

}
void ImproperHarmonic::takeValues(ImproperHarmonic &other) {
    k = other.k;
    thetaEq = other.thetaEq;
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
