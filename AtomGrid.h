#ifndef ATOMGRID_H
#define ATOMGRID_H
#include "Python.h"
#include "Grid.h"
#include "Bounds.h"
#include "Atom.h"
#include <math.h>
#include "OffsetObj.h"
#include <iostream>
#include <assert.h>
#include <vector>
#include <boost/shared_ptr.hpp>
#include "Mod.h"
#include "globalDefs.h"
#include "boost_for_export.h"
class State;
using namespace std;
using namespace boost;

void export_AtomGrid();

class AtomGrid : public Grid<Atom *> {
	
	void populateGrid() {
	}
	void appendNeighborList(Atom &, OffsetObj<Atom **> &, Vector, double);
	void appendNeighborListSelfCheck(Atom &a, OffsetObj<Atom **> &gridSqr, Vector boundsTrace, double neighCutSqr);
	//bounds oz should be set in Bounds
	void setNeighborSquares();
	void enforcePeriodicUnskewed(Bounds); //write a wrapper func if want to call externally b/c skew
	State *state;
    void init(double dx_, double dy_, double dz_);
    Bounds boundsOnGridding;
	public:
		vector<vector<OffsetObj<Atom **> > > neighborSquaress;
		double angX, angY;
		//rescale should probably go in here
		void updateAtoms(vector<Atom *> &);
        bool isSet;
        GridGPU makeGPU();
        AtomGrid() : isSet(false) {};
		AtomGrid(State *state_, double dx_, double dy_, double dz_);
		AtomGrid(SHARED(State) state_, double dx_, double dy_, double dz_);
		void enforcePeriodic(Bounds);
		/*
			Vector trace = state->bounds->trace;
			Vector attemptDDim = Vector(dx_, dy_, dz_);
			VectorInt nGrid = trace / attemptDDim; //so rounding to bigger grid
			
			Vector actualDDim = trace / nGrid; 
			//making grid that is exactly size of box.  This way can compute offsets easily from Grid that doesn't have to deal with higher-level stuff like bounds	
			is2d = state->is2d;
			ns = nGrid;
			ds = actualDDim;
			os = state->bounds->lo;
			dsOrig = actualDDim;
			fillVal = (Atom *) NULL;
			fillVals();
			saveRaw();
			setNeighborSquares();
		};
		*/
        void assignBondOffsets(vector<Bond> &, Bounds);
        bool adjustForChangedBounds();
		void periodicBoundaryConditions();
		void periodicBoundaryConditions(double);
        void buildNeighborlists(double);
		void buildNeighborlistRedund(Atom *, double);
		void resizeToStateBounds(bool);//these take new dimensions
		void deleteNeighbors();
		void populateLists();

};


#endif
