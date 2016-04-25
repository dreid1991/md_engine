#pragma once
#ifndef IMPROPER_H
#define IMPROPER_H

#include "globalDefs.h"
#include "Atom.h"

#include "cutils_math.h"
#include <boost/variant.hpp>
#include <array>
void export_Impropers();
class Improper {
    public:

        std::array<int, 4> ids;
        int type;
};



class ImproperHarmonic : public Improper {
    public:
        double k;
        double thetaEq;
        ImproperHarmonic(Atom *a, Atom *b, Atom *c, Atom *d, double k, double thetaEq, int type_=-1);
        ImproperHarmonic(double k, double thetaEq, int type_=-1);
        ImproperHarmonic(){};
        void takeParameters(ImproperHarmonic &);
        void takeIds(ImproperHarmonic &);
    
};

class ImproperHarmonicGPU {
    public:
        int ids[4];
        int myIdx;
        float k;
        float thetaEq;
        void takeParameters(ImproperHarmonic &);
        void takeIds(ImproperHarmonic &);


};
typedef boost::variant<
	ImproperHarmonic, 
    Improper	
> ImproperVariant;

#endif
