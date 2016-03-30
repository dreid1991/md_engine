#ifndef ANGLE_H
#define ANGLE_H
#include "globalDefs.h"
#include "Atom.h"

#include "cutils_math.h"

class Angle {
    public:
        //going to try storing by id instead.  Makes preparing for a run less intensive
        Atom *atoms[3];
};



class AngleHarmonic : public Angle {
    public:
        double thetaEq;
        double k;
        int type;
        AngleHarmonic(Atom *a, Atom *b, Atom *c, double k_, double thetaEq_, int type_=1);
        AngleHarmonic(double k_, double thetaEq_, int type_=-1);
        AngleHarmonic(){};
        void takeValues(AngleHarmonic &);
    
};

class AngleHarmonicGPU {
    public:
        int ids[3];
        int myIdx;
        float k;
        float thetaEq;
        void takeIds(int *);
        void takeValues(AngleHarmonic &);


};

#endif
