#pragma once
#ifndef MOLECULE_H

#include <vector>
#include "Vector.h"

class State;

void export_Molecule();

class Molecule {
private:
    State *state;
public:
    std::vector<int> ids;
    Molecule(State *, std::vector<int> &ids_);
    void translate(Vector &);
    void rotate(Vector &around, Vector &axis, double theta);
    Vector COM();
};




#endif
