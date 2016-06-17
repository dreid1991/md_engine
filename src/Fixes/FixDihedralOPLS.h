#pragma once
#ifndef FIXDIHEDRALOPLS_H
#define FIXDIHEDRALOPLS_H

#include <boost/python.hpp>

#include "FixPotentialMultiAtom.h"
#include "Dihedral.h"
#include "VariantPyListInterface.h"

void export_FixDihedralOPLS();

class FixDihedralOPLS : public FixPotentialMultiAtom<DihedralVariant, DihedralOPLS, Dihedral, DihedralGPU, DihedralOPLSType, 4> {

public:
    VariantPyListInterface<DihedralVariant, DihedralOPLS> pyListInterface;
    //DataSet *eng;
    //DataSet *press;

    FixDihedralOPLS(boost::shared_ptr<State> state_, std::string handle);

    void compute(bool);

    void createDihedral(Atom *, Atom *, Atom *, Atom *, double, double, double, double, int);
    void createDihedralPy(Atom *, Atom *, Atom *, Atom *, boost::python::list, int);
    void setDihedralTypeCoefs(int, boost::python::list);
    
    bool readFromRestart(pugi::xml_node restData);
    //std::vector<pair<int, std::vector<int> > > neighborlistExclusions();

};

#endif
