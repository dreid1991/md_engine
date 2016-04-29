#pragma once
#ifndef DIHEDRAL_H
#define DIHEDRAL_H
#include "globalDefs.h"
#include "Atom.h"

#include "cutils_math.h"
#include <boost/variant.hpp>
#include <array>
void export_Dihedrals();
class Dihedral{
    public:
        //going to try storing by id instead.  Makes preparing for a run less intensive
        std::array<int, 4> ids;
        int type;
};



class DihedralOPLS : public Dihedral {
    public:
        std::array<double, 4> coefs;
        DihedralOPLS(Atom *a, Atom *b, Atom *c, Atom *d, double coefs_[4], int type_);
        DihedralOPLS(double coefs_[4], int type_);
        void takeParameters(DihedralOPLS &);
        void takeIds(DihedralOPLS &);
        DihedralOPLS(){};
    
};

class DihedralOPLSGPU {
    public:
        int ids[4];
        int myIdx;
        float coefs[4];
        void takeParameters(DihedralOPLS &);
        void takeIds(DihedralOPLS &);


};

typedef boost::variant<
	DihedralOPLS, 
    Dihedral	
> DihedralVariant;
#endif
