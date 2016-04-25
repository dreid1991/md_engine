#pragma once
#ifndef ANGLE_H
#define ANGLE_H
#include "globalDefs.h"
#include "Atom.h"

#include "cutils_math.h"
#include <boost/variant.hpp>
#include <array>

void export_AngleHarmonic();

class Angle {
    public:
        //going to try storing by id instead.  Makes preparing for a run less intensive
        int type;
        std::array<int, 3> ids;
};



class AngleHarmonic : public Angle {
    public:
        double thetaEq;
        double k;
        AngleHarmonic(Atom *a, Atom *b, Atom *c, double k_, double thetaEq_, int type_=1);
        AngleHarmonic(double k_, double thetaEq_, int type_=-1);
        AngleHarmonic(){};
        void takeParameters(AngleHarmonic &);
        void takeIds(AngleHarmonic &);
    
};

class AngleHarmonicGPU {
    public:
        int ids[3];
        int myIdx;
        float k;
        float thetaEq;
        void takeParameters(AngleHarmonic &);
        void takeIds(AngleHarmonic &);


};
// lets us store a list of vectors to any kind of angles we want
typedef boost::variant<
	AngleHarmonic, 
    Angle	
> AngleVariant;
#endif
