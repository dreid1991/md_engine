#pragma once
#ifndef MOD_H
#define MOD_H

#include <assert.h>
#include <vector>
#include <random>

#include "Atom.h"
#include "globalDefs.h"
#include "Vector.h"

class State;
class Bond;
class Angle;

// mods are just tools.  they are called, do their job, and then go away.  They
// cannot depend on fixes

class ModPythonWrap { };

// convention: if you want good neighbors, do it yourself
namespace Mod {
    /*
    bool deleteBonds(SHARED(State) state, std::string groupHandle);
    void bondWithCutoff(SHARED(State) state, std::string groupHandle, num sigMultCutoff, num k);
    std::vector<int> computeNumBonds(SHARED(State) state, std::string groupHandle);
    std::vector<num> computeBondStresses(SHARED(State));
    bool singleSideFromVectors(std::vector<Vector> &vectors, bool is2d, Vector &trace);
    bool atomSingleSide(Atom *a, std::vector<Bond> &bonds, bool is2d, Vector &trace);
    std::vector<int> atomsSingleSide(SHARED(State), std::vector<Atom *> &atoms, std::vector<Bond> &bonds);
    bool deleteAtomsWithBondThreshold(SHARED(State), std::string, int thresh, int polarity);
    bool deleteAtomsWithSingleSideBonds(SHARED(State), std::string groupHandle);
    bool setZValue(SHARED(State), num neighThresh, const num target, const num tolerance, const num kBond, const bool display);
    num computeZ(SHARED(State), std::string groupHandle);
    */

    // HEY JUST COPY FROM MAIN FOLDER
    __global__ void unskewAtoms(real4 *xs, int nAtoms, real3 xOrig, real3 yOrig, real3 lo);
    __global__ void skewAtomsFromZero(real4 *xs, int nAtoms, real3 xFinal, real3 yFinal, real3 lo);

    template<bool>
    __global__ void scaleSystem_cu(real4 *xs, int nAtoms, real3 lo, real3 rectLen, real3 scaleBy,int *idToIdxs, 
                                         uint *notRigidBody);
    template<bool>
    __global__ void scaleSystemGroup_cu(real4 *xs, int nAtoms, real3 lo, real3 rectLen, real3 scaleBy, uint32_t groupTag, real4 *fs, int *idToIdxs, uint *notRigidBody);
    void scaleSystem(State *, real3 scaleBy, uint32_t groupTag=1);
    //__global__ void skewAtomsFromZero(cudaSurfaceObject_t xs, real4 xFinal, real4 yFinal);
    //__global__ void skewAtoms(cudaSurfaceObject_t xs, real4 xOrig, real4 xFinal, real4 yOrig, real4 yFinal);
    //__global__ void skew(SHARED(State), Vector);

    // CPU versions
    void unskewAtoms(std::vector<Atom> &atoms, Vector xOrig, Vector yOrig);
    void skewAtomsFromZero(std::vector<Atom> &atoms, Vector xFinal, Vector yFinal);
    void skewAtoms(std::vector<Atom> &atoms, Vector xOrig, Vector xFinal, Vector yOrig, Vector yFinal);
    void skew(SHARED(State), Vector);
    void scaleAtomCoords(SHARED(State) state, std::string groupHandle, Vector around, Vector scaleBy);
    void scaleAtomCoords(State *state, std::string groupHandle, Vector around, Vector scaleBy);

    void FDotR(State *state);
    inline Vector periodicWrap(Vector v, Vector sides[3], Vector offset) {
        for (int i=0; i<3; i++) {
            v += sides[i] * offset[i];
        }
        return v;
    }
}

#endif
