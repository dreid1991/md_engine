#ifndef DIHEDRAL_H
#define DIHEDRAL_H
#include "globalDefs.h"
#include "Atom.h"

#include "cutils_math.h"

class Dihedral{
    public:
        //going to try storing by id instead.  Makes preparing for a run less intensive
        Atom *atoms[4];
};



class DihedralOPLS : public Dihedral {
    public:
        double coefs[4];
        DihedralOPLS(Atom *a, Atom *b, Atom *c, Atom *d, double coefs_[4]);
    
};

class DihedralOPLSGPU {
    public:
        int ids[4];
        int myIdx;
        float coefs[4];
        void takeIds(int *);
        void takeValues(DihedralOPLS &);


};
#endif
