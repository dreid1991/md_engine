#pragma once
#ifndef ATOM_H
#define ATOM_H

#include <vector>

#include "Vector.h"
#include "OffsetObj.h"

class Atom;

typedef OffsetObj<Atom *> Neighbor;
typedef std::vector<Atom *> atomlist;

void export_Atom();
void export_Neighbor();

class Atom {
public:
    Vector pos;
    Vector vel;
    Vector force;
    Vector forceLast;
    //Vector posAtNeighborListing; // can do this elsewhere

    Atom *next;
    double mass;
    double q;
    int type;  // do this here, since otherwise would have to get it from some other array in the heap
    int id;
    uint32_t groupTag;
    std::vector<Neighbor> neighbors;
    bool isChanged;

    Atom()
      : mass(-1), id(0), groupTag(1)
    {   }

    Atom(int type_, int id_)
      : next(nullptr), mass(-1), q(0),
        type(type_), id(id_), groupTag(1)
    {   }

    Atom(Vector pos_, int type_, int id_)
      : pos(pos_), next(nullptr), mass(-1), q(0),
        type(type_), id(id_), groupTag(1)
    {   }

    Atom(Vector pos_, int type_, int id_, double mass_, double q_)
      : pos(pos_), next(nullptr), mass(mass_), q(q_),
        type(type_), id(id_), groupTag(1)
    {   }

    bool operator==(const Atom &other) {
        return id == other.id;
    }
    bool operator!=(const Atom &other) {
        return id != other.id;
    }

    double kinetic() {
        return 0.5 * mass * vel.lenSqr();
    }

    void setPos(Vector pos_);

    Vector getPos();

};

#endif
