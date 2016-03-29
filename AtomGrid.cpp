#include "AtomGrid.h"
#include "State.h"
#include "Mod.h"


void AtomGrid::init(double dx_, double dy_, double dz_) {
    if (not state->bounds.isSet) {
        cout << "Trying to start grid without setting bounds!" << endl;
        assert(state->bounds.isSet);
    }
    Vector trace = state->bounds.trace;
	Vector attemptDDim = Vector(dx_, dy_, dz_);
	VectorInt nGrid = trace / attemptDDim; //so rounding to bigger grid
	Vector actualDDim = trace / nGrid; 
	//making grid that is exactly size of box.  This way can compute offsets easily from Grid that doesn't have to deal with higher-level stuff like bounds	
    cout << state->is2d << endl;
	is2d = state->is2d;
	ns = nGrid;
	ds = actualDDim;
	os = state->bounds.lo;
	if (is2d) {
		ns[2]=1;
		ds[2]=1;
		assert(os[2]==-.5);
	}
	dsOrig = actualDDim;
    boundsOnGridding = state->bounds;
	fillVal = (Atom *) NULL;
	fillVals();
	saveRaw();
	setNeighborSquares();
}

AtomGrid::AtomGrid(SHARED(State) state_, double dx_, double dy_, double dz_) : state(state_.get()), isSet(true) {
    init(dx_, dy_, dz_);
}
AtomGrid::AtomGrid(State *state_, double dx_, double dy_, double dz_) : state(state_), isSet(true) {
    init(dx_, dy_, dz_);
    }


void AtomGrid::appendNeighborList(Atom &a, OffsetObj<Atom **> &gridSqr, Vector boundsTrace, double neighCutSqr) {
	if (*gridSqr.obj != (Atom *) NULL) {
		Vector offset = gridSqr.offset;
		Atom *current;
		for (current = *gridSqr.obj; current != (Atom *)NULL; current = current->next) {
			if (a.pos.distSqr(current->pos + offset*boundsTrace) <= neighCutSqr) {
				a.neighbors.push_back(Neighbor(current, offset));	
			}
		}
	}
}

bool AtomGrid::adjustForChangedBounds() {
    auto isConsistent = [] (Bounds b) {
        Vector lo = b.lo;
        if (b.isSkewed()) {
            for (int i=0; i<3; i++) {
                lo += b.sides[i];
            }
            return lo==b.hi;
        } 
        return true; //if not skewed, just go with lo, hi
         
    };
    if (state->bounds != boundsOnGridding) {
        bool toReturn = true;
        Vector changeInTrace = (state->bounds.hi-state->bounds.lo) - (boundsOnGridding.hi-boundsOnGridding.lo);
        if (state->bounds.lo > boundsOnGridding.lo) {
            cout << "Warning - you shrank the lo vector of bounds in at least one dimension.  Gridding may fail" << endl;
            toReturn = false;
        }
        if (state->bounds.hi < boundsOnGridding.hi) {
            cout << "Warning - you shrank the hi vector of bounds in at least one dimension.  Gridding may fail" << endl;
            toReturn = false;
        }
        if (state->bounds.isSkewed()) {
            if (not isConsistent(state->bounds)) {
                //going to try recovery
                for (int i=0; i<3; i++) {
                    state->bounds.sides[i][i] += changeInTrace[i];
                }
                if (not isConsistent(state->bounds)) {
                    cout << "Bounds changed such that lo, hi, and sides, are not consistent" << endl;

                    assert(false);
                }

            }
        } else {
            state->bounds.setSides();
        }
 

        resizeToStateBounds(false);
        return toReturn;
    }
    return true;
}

void AtomGrid::appendNeighborListSelfCheck(Atom &a, OffsetObj<Atom **> &gridSqr, Vector boundsTrace, double neighCutSqr) {
	if (*gridSqr.obj != (Atom *) NULL) {
		Vector offset = gridSqr.offset;
		Atom *current;
		for (current = *gridSqr.obj; current != (Atom *)NULL; current = current->next) {
			if (&a != current and a.pos.distSqr(current->pos + offset*boundsTrace) <= neighCutSqr) {
				a.neighbors.push_back(Neighbor(current, offset));	
			}
		}
	}
}




GridGPU AtomGrid::makeGPU() {
    return GridGPU(state, ds.asFloat3(), dsOrig.asFloat3(), os.asFloat3(), ns.asInt3());
}

void AtomGrid::enforcePeriodic(Bounds bounds) {
	vector<Atom> &atoms = state->atoms;
	if (bounds.sides[0][1] or bounds.sides[1][0]) {
		Mod::unskewAtoms(atoms, bounds.sides[0], bounds.sides[1]);
	}

	enforcePeriodicUnskewed(bounds.unskewed());

	if (bounds.sides[0][1] or bounds.sides[1][0]) {
		Mod::skewAtomsFromZero(atoms, bounds.sides[0], bounds.sides[1]);
	}
}


void AtomGrid::enforcePeriodicUnskewed(Bounds bounds) { //make it so doesn't loop in finite dimensions
	vector<Atom> &atoms = state->atoms;
	Vector lo = bounds.lo;
	Vector hi = bounds.hi;
	Vector trace = bounds.trace;
	for (Atom &a : atoms) {
		Vector prev = a.pos;
		for (int i=0; i<3; i++) {
			if (a.pos[i] < lo[i]) {
				a.pos[i] += trace[i];
			} else if (a.pos[i] >= hi[i]) {
				a.pos[i] -= trace[i];
			}
			//IF YOU GET MYSTERIOUS CRASHES, THERE MAY BE FLOATING POINT ERROR WHERE ADDING/SUBTRACTING TRACE PUTS IT SLIGHTLY OFF OF THE GRID
		}
		if (!bounds.atomInBounds(a)) {
            cout << "Error: atom out of bonds" << endl;
			cout << "position before trying to loop " << prev.asStr() << endl;
			cout << "turn " << state->turn << endl;
			cout << "Atom id " << a.id << " moved more than one box length since building neighbor lists.  Program about to quit.  Consider decreasing your neighboring interval" << endl;
			assert(false);
		}
	}
}

void AtomGrid::setNeighborSquares() {
	Bounds bounds = state->bounds;
	bool periodic[3];
	for (int i=0; i<3; i++) {
		periodic[i] = state->periodic[i];
	}
	neighborSquaress = vector< vector< OffsetObj<Atom **> > >(); 
	neighborSquaress.reserve(raw.size());
	for (int i=0; i<ns[0]; i++) {
		for (int j=0; j<ns[1]; j++) {
			for (int k=0; k<ns[2]; k++) {
				int coord[3];
				coord[0] = i; coord[1] = j; coord[2] = k;
				neighborSquaress.push_back(getNeighbors(coord, periodic, bounds.trace));
			}
		}
	}
	//okay, just need to make sure that raw address stays constant and this should work.  It should... stays same size
}


void AtomGrid::buildNeighborlists(double neighCut) {
    cout << "BUILDING" << endl;
    vector<Atom> &atoms = state->atoms;
	Vector boundsTrace = state->bounds.trace;
	bool periodic[3];
	for (int i=0; i<3; i++) {
		periodic[i] = state->periodic[i];
	}
	assert(not (state->is2d and periodic[2]));


	double neighCutSqr = neighCut*neighCut;
	for (unsigned int i=0; i<atoms.size(); i++) {
		Atom &a = atoms[i];
		a.neighbors.erase(a.neighbors.begin(), a.neighbors.end()); 
	}
	reset();
	/*
	*using looping values, make list of squares that corresponds to the neighbors for each square.
	 Then for each atom, add atoms by following each linked list and appening those within rcut
	*/ 
	

	//doing assumed newton, don't have double counted neigborlists
	for (Atom &a : atoms) {
		int idx = idxFromPos(a.pos);		
		Atom **neighborSquare = &(*this)(a.pos);
		OffsetObj<Atom **> selfSquare = OffsetObj<Atom **>(neighborSquare, Vector(0, 0, 0));
		appendNeighborList(a, selfSquare, boundsTrace, neighCutSqr);
		a.next = *neighborSquare;
		*neighborSquare = &a;
		vector<OffsetObj<Atom **> > &neighborSquares = neighborSquaress[idx];
		for (OffsetObj<Atom **> &neighborSquare : neighborSquares) {
			appendNeighborList(a, neighborSquare, boundsTrace, neighCutSqr);	
		}
	}
}

/*
void AtomGrid::assignBondOffsets(vector<Bond> &bonds, Bounds bounds) {
    //okay, things are unskewed at this point
    cout << "STOP ASSIGNING BOND OFFSETS" << endl;
    Vector half = bounds.trace / (double) 2;
    for (Bond &b : bonds) {
        Vector offset;
        Vector dPos = b.atoms[1]->pos - b.atoms[0]->pos;
        for (int i=0; i<3; i++) {
            if (dPos[i] > half[i]) {
				offset[i] = -1;
			} else if (dPos[i] < -half[i]) {
				offset[i] = 1;
			}
        }
        //b.offset = offset;
    }

}
*/
void AtomGrid::periodicBoundaryConditions(double neighCut) {
	Bounds &unchanged = state->bounds;
	Bounds b = state->bounds.unskewed();
	vector<Atom> &atoms = state->atoms;

    bool isSkew = state->bounds.isSkewed();
    if (isSkew) {
        Mod::unskewAtoms(atoms, unchanged.sides[0], unchanged.sides[1]);
    }
	enforcePeriodicUnskewed(b);
    if (state->buildNeighborlists) {
        buildNeighborlists(neighCut);
    }
  //  assignBondOffsets(state->bonds, b);
    if (isSkew) {
        Mod::skewAtomsFromZero(atoms, unchanged.sides[0], unchanged.sides[1]);
    }
}
void AtomGrid::periodicBoundaryConditions() { //grid size must be >= 2*neighCut
	double neighCut = state->rCut + state->padding;
	periodicBoundaryConditions(neighCut);
}




//untested
void AtomGrid::populateLists() {
	reset();
	vector<Atom> &atoms = state->atoms;
	for (Atom &a : atoms) { //this is going to be super slow b/c linked list, but I'm really not doing it often.  Could improve by specializing w/ vector stuff
		int idx = idxFromPos(a.pos);
		a.next = raw[idx];
		raw[idx] = &a;
	}
}
void AtomGrid::buildNeighborlistRedund(Atom *a, double neighCut) {
	Vector boundsTrace = state->bounds.trace;
	double neighCutSqr = neighCut * neighCut;
	a->neighbors.erase(a->neighbors.begin(), a->neighbors.end()); 
	Vector pos = a->pos;

	OffsetObj<Atom **> selfSqr;
	selfSqr.obj = &(*this)(a->pos);
	selfSqr.offset = Vector(0, 0, 0);
	appendNeighborListSelfCheck(*a, selfSqr, boundsTrace, neighCutSqr); 

	for (OffsetObj<Atom **> neighborSqr : neighborSquaress[idxFromPos(pos)]) {
		appendNeighborList(*a, neighborSqr, boundsTrace, neighCutSqr); 
	}
}




void AtomGrid::resizeToStateBounds(bool scaleAtomCoords) {
    //DEALS ONLY WITH UNSKEWED FOR NOW
    //
    //
	//I *think* this will work with skewed bounds, untested
    //sooo... this is just called once bounds have been set, could be cleaned up

	Bounds &boundsCur = boundsOnGridding;
    Bounds &boundsNew = state->bounds;


    Bounds boundsNewUnskewed = boundsNew.unskewed();
    Vector traceNew = boundsNewUnskewed.trace;
    Vector traceOld = boundsCur.trace;
	if (state->is2d) {
		assert(traceNew[2] == 1);
	}
	//keeping it centered around same origin


	//now to handle grid
	//okay, so let's keep d xyz constant, and just change the doubleber of grid cells when necessary.  Never necessary when shrinking, but don't want to be sloppy.  

	Vector dsProposed = traceNew / ns;
	Vector nsNew = ns;
	int maxDimCheck;
	if (state->is2d) {
		maxDimCheck = 2;
	} else {
		maxDimCheck = 3;
	}
	Bounds unskewed = boundsCur.unskewed();
	double gridSqrMin = state->rCut + state->padding;
	for (int i=0; i<maxDimCheck; i++) {
		bool haveResized = false;
		//eeh, could refine more, but this works
		while (dsProposed[i] < gridSqrMin || dsProposed[i] < .7 * dsOrig[i]) {
			nsNew[i]--;
			dsProposed[i] = unskewed.trace[i] / nsNew[i];
			haveResized = true;
		}
		if (not haveResized) {
			while (dsProposed[i] > 1.3 * dsOrig[i]) {
				nsNew[i]++;
				dsProposed[i] = unskewed.trace[i] / nsNew[i];
			}
		}
	}
	os = boundsNew.lo;
	ds = dsProposed;
	if (ns != nsNew) {
        //this should work with skew, since it doesn't force reneighboring
		ns = nsNew;
		fillVals();
		saveRaw();
		setNeighborSquares();
	}
	if (scaleAtomCoords) {
		Vector center = (boundsNew.lo + boundsNew.hi) / (double) 2;
		Vector scaleBy = traceNew / traceOld;
		Mod::scaleAtomCoords(state, "all", center, scaleBy);
	}
    boundsOnGridding = state->bounds;
}


/*
void AtomGrid::shear(double angleX, double angleY) {
	State *raw = state.get();
	Bounds &b = *raw->Bounds;
	vector<Atom> &atoms = raw->atoms;
	
}
*/
void AtomGrid::deleteNeighbors() {
	for (Atom &a : state->atoms) {
		a.neighbors.erase(a.neighbors.begin(), a.neighbors.end()); 
	}
}
void export_AtomGrid() {
    class_<AtomGrid>("AtomGrid", init<SHARED(State), double, double, double>(args("state", "dx", "dy", "dz")))
        .def_readwrite("os", &AtomGrid::os)
        .def_readwrite("ds", &AtomGrid::ds)
        .def_readwrite("ns", &AtomGrid::ns)
        ;

}
