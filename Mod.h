#pragma once
#ifndef MOD_H
#define MOD_H
class State;
class Bond;
class Angle;
using namespace std;
#include "globalDefs.h"
#include "Vector.h"
#include <vector>
#include <assert.h>
#include <random>
#include "Atom.h"
//mods are just tools.  they are called, do their job, and then go away.  They cannot depend on fixes

class ModPythonWrap {};
//convention: if you want good neighbors, do it yourself
namespace Mod {
    /*
	bool deleteBonds(SHARED(State) state, string groupHandle);
	void bondWithCutoff(SHARED(State) state, string groupHandle, num sigMultCutoff, num k);
	vector<int> computeNumBonds(SHARED(State) state, string groupHandle);
	vector<num> computeBondStresses(SHARED(State));
	bool singleSideFromVectors(vector<Vector> &vectors, bool is2d, Vector &trace);
	bool atomSingleSide(Atom *a, vector<Bond> &bonds, bool is2d, Vector &trace); 
	vector<int> atomsSingleSide(SHARED(State), vector<Atom *> &atoms, vector<Bond> &bonds);
	bool deleteAtomsWithBondThreshold(SHARED(State), string, int thresh, int polarity);
	bool deleteAtomsWithSingleSideBonds(SHARED(State), string groupHandle);
	bool setZValue(SHARED(State), num neighThresh, const num target, const num tolerance, const num kBond, const bool display);
	num computeZ(SHARED(State), string groupHandle);
    */
    

    //HEY JUST COPY FROM MAIN FOLDER
	vector<vector<Bond *> > musterBonds(State *state, vector<Bond *> &bonds);
	vector<vector<Angle *> > musterAngles(State *state, vector<Angle *> &angles);
	__global__ void unskewAtoms(float4 *xs, int nAtoms, float3 xOrig, float3 yOrig, float3 lo);
	__global__ void skewAtomsFromZero(float4 *xs, int nAtoms, float3 xFinal, float3 yFinal, float3 lo);
	//__global__ void skewAtomsFromZero(cudaSurfaceObject_t xs, float4 xFinal, float4 yFinal);
    //__global__ void skewAtoms(cudaSurfaceObject_t xs, float4 xOrig, float4 xFinal, float4 yOrig, float4 yFinal);
    //__global__ void skew(SHARED(State), Vector);


    //CPU versions
	void unskewAtoms(vector<Atom> &atoms, Vector xOrig, Vector yOrig);
	void skewAtomsFromZero(vector<Atom> &atoms, Vector xFinal, Vector yFinal);
    void skewAtoms(vector<Atom> &atoms, Vector xOrig, Vector xFinal, Vector yOrig, Vector yFinal);
    void skew(SHARED(State), Vector);
	void scaleAtomCoords(SHARED(State) state, string groupHandle, Vector around, Vector scaleBy);
	void scaleAtomCoords(State *state, string groupHandle, Vector around, Vector scaleBy);
    inline Vector periodicWrap(Vector v, Vector sides[3], Vector offset) {
        for (int i=0; i<3; i++) {
            v += sides[i] * offset[i];
        }
        return v;

    }
}

#endif


