#pragma once
#include "globalDefs.h"
#include "Atom.h"

#include "cutils_math.h"

class Improper {
    public:
        Atom *atoms[4];
};



class ImproperHarmonic : public Improper {
    public:
        double k;
        double thetaEq;
        ImproperHarmonic(Atom *a, Atom *b, Atom *c, Atom *d, double k, double thetaEq);
    
};

class ImproperHarmonicGPU {
    public:
        int ids[4];
        int myIdx;
        float k;
        float thetaEq;
        void takeIds(int *);
        void takeValues(ImproperHarmonic &);


};
