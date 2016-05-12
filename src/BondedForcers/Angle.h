#pragma once
#ifndef ANGLE_H
#define ANGLE_H
#include "globalDefs.h"
#include "Atom.h"

#include "cutils_math.h"
#include <boost/variant.hpp>
#include <array>
class AngleHarmonic;
void export_AngleHarmonic();

class Angle {
    public:
        //going to try storing by id instead.  Makes preparing for a run less intensive
        int type;
        std::array<int, 3> ids;
        void takeIds(Angle *);
};

class AngleHarmonicType {
    public:
        float k;
        float thetaEq;
        AngleHarmonicType(AngleHarmonic *);
        AngleHarmonicType(){};
        bool operator==(const AngleHarmonicType &) const;
};

class AngleHarmonic : public Angle, public AngleHarmonicType {
    public:
        AngleHarmonic(Atom *a, Atom *b, Atom *c, double k_, double thetaEq_, int type_=1);
        AngleHarmonic(double k_, double thetaEq_, int type_=-1);
        AngleHarmonic(){};
        int type;
    
};

//for forcer maps
namespace std {
    template<> struct hash<AngleHarmonicType> {
        size_t operator() (AngleHarmonicType const& ang) const {
            size_t seed = 0;
            boost::hash_combine(seed, ang.k);
            boost::hash_combine(seed, ang.thetaEq);
            return seed;
        }
    };


}
class AngleGPU {
    public:
        int ids[3];
        uint32_t type; //myIdx (which atom in these three we're actually calcing the for for) is stored in two left-most bits
        void takeIds(Angle *);


};
// lets us store a list of vectors to any kind of angles we want
typedef boost::variant<
	AngleHarmonic, 
    Angle	
> AngleVariant;
#endif
