#pragma once
#ifndef IMPROPER_H
#define IMPROPER_H

#include "globalDefs.h"
#include "Atom.h"

#include "cutils_math.h"
#include <boost/variant.hpp>

class Improper {
    public:
        Atom *atoms[4];
        int type;
};



class ImproperHarmonic : public Improper {
    public:
        double k;
        double thetaEq;
        ImproperHarmonic(Atom *a, Atom *b, Atom *c, Atom *d, double k, double thetaEq, int type_=-1);
        ImproperHarmonic(double k, double thetaEq, int type_=-1);
        ImproperHarmonic(){};
        void takeValues(ImproperHarmonic &);
    
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
typedef boost::variant<
	ImproperHarmonic, 
    Improper	
> ImproperVariant;

#endif
