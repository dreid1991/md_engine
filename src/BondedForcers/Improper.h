#pragma once
#ifndef IMPROPER_H
#define IMPROPER_H

#include "globalDefs.h"
#include "Atom.h"

#include "cutils_math.h"
#include <boost/variant.hpp>
#include <boost/functional/hash.hpp>
#include <array>
void export_Impropers();
class Improper {
    public:

        std::array<int, 4> ids;
        int type;
        void takeIds(Improper *);
};



class ImproperHarmonic : public Improper {
    public:
        double k;
        double thetaEq;
        ImproperHarmonic(Atom *a, Atom *b, Atom *c, Atom *d, double k, double thetaEq, int type_=-1);
        ImproperHarmonic(double k, double thetaEq, int type_=-1);
        ImproperHarmonic(){};
    
};

class ImproperHarmonicType {
    public:
        float thetaEq;
        float k;
        ImproperHarmonicType(ImproperHarmonic *);
        ImproperHarmonicType(){}; //for hashing, need default constructor, == operator, and std::hash function
        bool operator==(const ImproperHarmonicType &) const;
};

class ImproperGPU{
    public:
        int ids[4];
        uint32_t type;
        void takeIds(Improper *);


};
//for forcer maps
namespace std {
    template<> struct hash<ImproperHarmonicType> {
        size_t operator() (ImproperHarmonicType const& imp) const {
            size_t seed = 0;
            boost::hash_combine(seed, imp.k);
            boost::hash_combine(seed, imp.thetaEq);
            return seed;
        }
    };


}
typedef boost::variant<
	ImproperHarmonic, 
    Improper	
> ImproperVariant;

#endif
