#pragma once
#ifndef BOND_H
#define BOND_H

#include <boost/variant.hpp>

#include "globalDefs.h"
#include "Atom.h"
#include <array>

/*! \brief Bond connecting atoms
 *
 * \link Atom Atoms\endlink can be connected by Bonds. Bonds are defined by a
 * potential depending on the separation of the two bonded \link Atom
 * Atoms\endlink. 
 */
class BondHarmonic;

class Bond {
    public:
        std::array<int, 2> ids;//!<atom ids
        int type; //!< Bond type
};


class BondHarmonicType {
public:
    float k;
    float rEq;
    BondHarmonicType(BondHarmonic *);
    BondHarmonicType(){};
    bool operator==(const BondHarmonicType &) const;
    std::string getInfoString();
};
//
//for forcer maps
namespace std {
    template<> struct hash<BondHarmonicType> {
        size_t operator() (BondHarmonicType const& bond) const {
            size_t seed = 0;
            boost::hash_combine(seed, bond.k);
            boost::hash_combine(seed, bond.rEq);
            return seed;
        }
    };
}

/*! \brief Bond with a harmonic potential (a spring)
 *
 * Bond with harmonic potential.
 *
 * \todo In LAMMPS k is, in fact, k/2. Specify this explicitely here.
 */





class BondHarmonic : public Bond, public BondHarmonicType {
	public:
        BondHarmonic(Atom *a, Atom *b, double k_, double rEq_, int type_=-1);
        BondHarmonic(double k_, double rEq_, int type_=-1); //is this constructor used?
        BondHarmonic(){};
        int type;
	std::string getInfoString();
};	

void export_BondHarmonic();


class __align__(16) BondGPU {
    public:
        int myId; //!< ID of this Atom
        int otherId; //!< ID of the other Atom in the Bond
        int type; //!bond type number
        void takeIds(Bond *b); //!copy myId, otherId out of Bond *

};


/*! \typedef Boost Variant for any bond */
typedef boost::variant<
	BondHarmonic, 
	Bond
> BondVariant;


#endif
