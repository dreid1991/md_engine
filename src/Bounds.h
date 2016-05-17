#pragma once
#ifndef BOUNDS_H
#define BOUNDS_H

#include <stdio.h>
#include <assert.h>

#include "Python.h"

#include "globalDefs.h"
#include "Atom.h"
#include "BoundsGeneric.h"
#include "BoundsGPU.h"

#include "boost_for_export.h"

void export_Bounds();

class State;

class Bounds : public BoundsGeneric {

public:
    State *state;
    bool isSet;

    Bounds()
        : isSet(false)
    { }

    Bounds(SHARED(State) state_, Vector lo_, Vector hi_)
        : BoundsGeneric(lo_, hi_), state(state_.get()), isSet(true)
    {   handle2d(); }

    Bounds(State *state_, Vector lo_, Vector hi_)
        : BoundsGeneric(lo_, hi_), state(state_), isSet(true)
    {   handle2d(); }

    Bounds(SHARED(State) state_, Vector lo_, Vector sides_[3])
        : BoundsGeneric(lo_, sides_), state(state_.get()), isSet(true)
    {   handle2d(); }

    Bounds(State *state_, Vector lo_, Vector sides_[3])
        : BoundsGeneric(lo_, sides_), state(state_), isSet(true)
    {   handle2d(); }

    Bounds(BoundsGPU &source)
        : state(nullptr), isSet(true)
    {   set(source); }

    bool operator==(const Bounds &other) {
        return ((lo-other.lo).abs() < VectorEps) &&
               ((hi-other.hi).abs() < VectorEps) &&
               ((sides[0]-other.sides[0]).abs() < VectorEps) &&
               ((sides[1]-other.sides[1]).abs() < VectorEps) &&
               ((sides[2]-other.sides[2]).abs() < VectorEps);
    }
    bool operator!=(const Bounds &other) {
        return !(*this == other);
    }

    Bounds copy() {
        return *this;
    }

    void set(BoundsGPU &b) {
        lo = Vector(b.lo);
        hi = lo;
        trace = Vector();
        for (int i=0; i<3; i++) {
            sides[i] = Vector(b.sides[i]);
            trace += sides[i];
        }
        hi += trace;
    }
    void set(Bounds &b) {
        lo = b.lo;
        hi = b.hi;
        for (int i=0; i<3; i++) {
            sides[i] = b.sides[i];
        }
        trace = b.trace;
    }
    void setSides() {
        trace = hi-lo;
        for (int i=0; i<3; i++) {
            Vector v = Vector(0, 0, 0);
            v[i] = trace[i];
            sides[i] = v;
        }
    }
    void setPython(Bounds &b) {
        set(b);
    }

    void handle2d();
    BoundsGPU makeGPU();
    bool atomInBounds(Atom &);
    double volume();

    bool skew(Vector v);
    double getSkew(int idx);
    double getSkewX();
    double getSkewY();
    Bounds unskewed();
    bool isSkewed();
    Vector minImage(Vector v);

};

//SHARED(Bounds) BoundsCreateSkew(  figure out how to create laters

#endif
