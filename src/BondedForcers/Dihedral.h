#pragma once
#ifndef DIHEDRAL_H
#define DIHEDRAL_H
#include "globalDefs.h"
#include "Atom.h"

#include "cutils_math.h"
#include <boost/variant.hpp>
#include <boost/functional/hash.hpp>
#include <array>
class DihedralOPLS;
void export_Dihedrals();
class Dihedral{
    public:
        //going to try storing by id instead.  Makes preparing for a run less intensive
        std::array<int, 4> ids;
        int type;
        void takeIds(Dihedral *);
	std::string getInfoString();
};

class DihedralOPLSType {
    public:
        float coefs[4];
        DihedralOPLSType(DihedralOPLS *);
        DihedralOPLSType(){};
        bool operator==(const DihedralOPLSType &) const;
	std::string getInfoString();
};



class DihedralOPLS : public Dihedral, public DihedralOPLSType {
    public:
        DihedralOPLS(Atom *a, Atom *b, Atom *c, Atom *d, double coefs_[4], int type_);
        DihedralOPLS(double coefs_[4], int type_);
        DihedralOPLS(){};
	std::string getInfoString();
};

class DihedralGPU {
    public:
        int ids[4];
        uint32_t type;
        void takeIds(Dihedral *);


};

//for forcer maps
namespace std {
    template<> struct hash<DihedralOPLSType> {
        size_t operator() (DihedralOPLSType const& dih) const {
            size_t seed = 0;
            for (int i=0; i<4; i++) {
                boost::hash_combine(seed, dih.coefs[i]);
            }
            return seed;
        }
    };


}

typedef boost::variant<
	DihedralOPLS, 
    Dihedral	
> DihedralVariant;
#endif
