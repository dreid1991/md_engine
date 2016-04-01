#include "Mod.h"
#include "State.h"

#include "Bond.h"
#include "Angle.h"



/*

void Mod::bondWithCutoff(SHARED(State) state, string groupHandle, num sigmaMultCutoff, num k) {
	state->makeReady();
	SHARED(AtomGrid) grid = state->grid;
	Bounds b = *state->bounds;
	vector<Atom> &atoms = state->atoms;
	bool changedBonds = false;
	state->redoNeighbors = true;
	SHARED(AtomParams) atomParams = state->atomParams;
	int numTypes = atomParams->numTypes;
	num sigmaMax = 0;
	for (int i=0; i<numTypes; i++) {
		for (int j=0; j<numTypes; j++) {
			sigmaMax = max(atomParams->sigmas[i][j], sigmaMax);
		}
	}
		
	grid->enforcePeriodic(b);
	grid->buildNeighborlists(sigmaMax * sigmaMultCutoff);
	int groupTag = state->groupTagFromHandle(groupHandle);
	vector<vector<num> > filledSigmas = atomParams->fillParams("sig");
	Vector trace = state->bounds->trace;
	for (Atom &a : atoms) {
		for (Neighbor n : a.neighbors) {
			Atom &b= *n.obj;
			if ((a.groupTag & groupTag) && (b.groupTag & groupTag)) {
				num distSqr = a.pos.distSqr(b.pos + n.offset * trace);
				num cutoff = filledSigmas[a.type][b.type] * sigmaMultCutoff;
				num cutoffSqr = cutoff * cutoff;
				if (distSqr < cutoffSqr) {
					state->addBond(&a, &b, k, sqrt(distSqr));
					changedBonds = true;
				}
			}
		}
	}
	if (changedBonds) {
		state->changedBonds = true;
	}
}

vector<num> Mod::computeBondStresses(SHARED(State) state) {
	state->makeReady();
	vector<num> stresses;
	stresses.reserve(state->bonds.size());
	Bounds bounds = *(state->bounds);
	Vector halfTrace = bounds.trace / (num) 2.0;
	for (Bond &b : state->bonds) {
		Vector pos[2];
		pos[0] = b.atoms[0]->pos;
		pos[1] = b.atoms[1]->pos;
		Vector dir = pos[1] - pos[0];
		for (int i=0; i<3; i++) {
			if (dir[i] > halfTrace[i]) {
				dir[i] -= bounds.trace[i];
			} else if (dir[i] < -halfTrace[i]) {
				dir[i] += bounds.trace[i];
			}
		}
		num dr = dir.len() - b.rEq;

		num stress = dr * dr * b.k * .5;
		stresses.push_back(stress);

	}
	return stresses;
}




vector<int> Mod::computeNumBonds(SHARED(State) state, string groupHandle) {

	state->makeReady();
	int groupTag = state->groupTagFromHandle(groupHandle);
	vector<int> numBonds;
	numBonds.reserve(state->atoms.size());
	for (int i=0, ii=state->atoms.size(); i<ii; i++) {
		numBonds.push_back(0);
	}
	vector<Atom> &atoms = state->atoms;
	vector<Bond> &bonds = state->bonds;
	for (Bond &b : bonds) {
		for (int i=0; i<2; i++) {
			Atom *a = b.atoms[i];
			int atomIdx = a - &(atoms[0]);
			numBonds[atomIdx]++;
		}
	}
	for (int i=atoms.size()-1; i>=0; i--) {
		if (not (atoms[i].groupTag & groupTag)) {
			numBonds.erase(numBonds.begin()+i, numBonds.begin()+i+1);
		}
	}
	return numBonds;
}


*/

vector<vector<Bond *> > Mod::musterBonds(State *state, vector<Bond *> &bonds) { //no group handle for now, can deal later
	vector<vector<Bond *> > bondss;
	Atom *begin = &state->atoms[0];
	bondss.reserve(state->atoms.size());
	for (int i=0, ii=state->atoms.size(); i<ii; i++) {
		bondss.push_back(vector<Bond *>());
	}

	for (Bond *b : bonds) {
		bondss[b->atoms[0] - begin].push_back(b);
		bondss[b->atoms[1] - begin].push_back(b);
	}
	//if (groupHandle == "all") {
	return bondss;
//	}
	//int groupTag = raw->groupTagFromHandle(group); //compute for all b/c it's n^2 otherwise.  Then just prune
	//make work with groups later
	//vector<Atom *> inGroup = LISTMAPREFTEST(Atom, Atom *, a, atoms, &a, a.groupTag & groupTag);
	//for (int i=inGroup.size()-1; i>=0; i++) {

	//}
}

//TRYING a different structure from bonds.
vector<vector<Angle *> > Mod::musterAngles(State *state, vector<Angle *> &angles) {
    cout << "REMOVE THIS WHEN YOU FEEL LIKE RECOMPILING" << endl;
    return vector<vector<Angle *> >();
}
/*
vector<vector<Angle *> > Mod::musterAngles(State *state, vector<Angle *> &angles) {
    vector<vector<Angle *> > angless;
    angless.reserve(state->atoms.size());
    //hmm, since IDs are pretty dense, can just make idx be id, and there will be none for from of them
    for (Angle *angle : angles) {
        int *atoms = angle->atoms;
        for (int i=0; i<3; i++) {
            while (angless.size() < atoms[i]) {
                angless.push_back(vector<Angle *>());
            }
        }
        for (int i=0; i<3; i++) {
            angless[atoms[i]].push_back(angle);
        }

    }
    return angless;

}
*/
/*
bool Mod::singleSideFromVectors(vector<Vector> &dirVectors, bool is2d, Vector &trace) {
	bool allOnSide = false;
	bool tooFew = (is2d && dirVectors.size() < 3) || (!is2d && dirVectors.size() < 4);
	for (Vector &v : dirVectors) {
		v.normalize();
	}
	if (not tooFew)	{
		vector<num> dirs;
		dirs.reserve(dirVectors.size());
		for (Vector &v : dirVectors) {
			dirs.push_back(atan2(v[1], v[0]));
		}
		sort(dirs.begin(), dirs.end());
		//now just check if any of the differences in angle are > 180 degrees
		for (unsigned int i=0; i<dirs.size()-1; i++) {
			if (dirs[i+1] - dirs[i] > M_PI) {
				allOnSide = true;
				break;
			}

		}
		if (dirs[0] + 2*M_PI - dirs[dirs.size()-1] > M_PI) {
			allOnSide = true;
		}

	}

	return allOnSide or tooFew;
}


bool Mod::atomSingleSide(Atom *a, vector<Bond> &bonds, bool is2d, Vector &trace) { 
	vector<Vector> dirVectors;
	for (Bond &b : bonds) {
		if (b.hasAtom(a)) {
			Vector fromAtom = b.vectorFrom(a, trace);
			dirVectors.push_back(fromAtom);

		}
	}
	return Mod::singleSideFromVectors(dirVectors, is2d, trace);
}

vector<int> Mod::atomsSingleSide(SHARED(State) state, vector<Atom *> &atoms, vector<Bond> &bonds) {
	Bounds bounds = *state->bounds;
	State *raw = state.get();
	vector<int> singleSide;
	bool is2d = raw->is2d;
	for (unsigned int i=0; i<atoms.size(); i++) {
		if (Mod::atomSingleSide(atoms[i], bonds, is2d, bounds.trace)) {
			singleSide.push_back(i);
		}
	}
	return singleSide;
}





bool Mod::deleteAtomsWithSingleSideBonds(SHARED(State) state, string groupHandle) {
	//I think this method only works in 2d.  Need to consider another axis in 3d
	State *raw = state.get();
	raw->makeReady();
	Bounds bounds = *state->bounds;
	int groupTag = raw->groupTagFromHandle(groupHandle);
	vector<Atom> &atoms = raw->atoms;
	vector<Bond> &bonds= raw->bonds;
	vector<Atom *> inGroup = LISTMAPREFTEST(Atom, Atom *, a, atoms, &a, groupTag & a.groupTag);
	
	vector<int> toRemove = Mod::atomsSingleSide(state, inGroup, bonds);
	//cout << "PUSHING ARTIFICIAL" << endl;
	//toRemove.push_back(5);
	sort(toRemove.begin(), toRemove.end(), greater<int>()); //sorting in reverse so idxs remain valid
	for (int i : toRemove) {
		raw->removeAtom(inGroup[i]);
	}
	return toRemove.size();


}




bool Mod::deleteAtomsWithBondThreshold(SHARED(State) state, string groupHandle, int thresh, int polarity) {
	State *raw = state.get();
	bool removedAny = false;
	vector<Atom> &atoms = raw->atoms;
	assert(polarity == 1 or polarity == -1);
	vector<int> numBonds = Mod::computeNumBonds(state, groupHandle);
	for (int i=numBonds.size()-1; i>=0; i--) {
		if (numBonds[i] * polarity >= thresh * polarity) {
			raw->removeAtom(&atoms[i]); //flags taken care of here
			removedAny = true;
		}
	}
	return removedAny;

}




bool Mod::deleteBonds(SHARED(State) state, string groupHandle) {
	State *raw = state.get();
	raw->makeReady();
	vector<Bond> &bonds = raw->bonds;
	bool removedAny = false;
	int groupTag = raw->groupTagFromHandle(groupHandle);
	for (int i=bonds.size()-1; i>=0; i--) {
		Bond &b = bonds[i];
		if ((b.atoms[0]->groupTag & groupTag) || (b.atoms[1]->groupTag & groupTag)) {
			state->removeBond(&b);
			removedAny = true;
		}
	}
	return removedAny;
}


void Mod::scaleAtomCoords(SHARED(State) state, string groupHandle, Vector around, Vector scaleBy) {
	//don't need to makeready for this one		
	//this should work for skewed bounds
	int groupTag = state->groupTagFromHandle(groupHandle);
	for (Atom &a : state->atoms) {
		if (a.groupTag & groupTag) {
			Vector diff = a.pos - around;
			diff *= scaleBy;
			a.pos = around + diff;
		}
	}

}


num Mod::computeZ(SHARED(State) state, string groupHandle) {
	State *raw = state.get();
	if (groupHandle=="all") {
		return 2 * raw->bonds.size() / (num) raw->atoms.size();
	}
	vector<int> numBonds = Mod::computeNumBonds(state, groupHandle);
	int groupTag = raw->groupTagFromHandle(groupHandle);
	int numTotal = accumulate(numBonds.begin(), numBonds.end(), 0);
	int numAtoms = 0;
	for (Atom &a : raw->atoms) {
		numAtoms += ((bool) (a.groupTag & groupTag));
	}
	return numTotal / (num) numAtoms;

}

bool Mod::setZValue(SHARED(State) state, num neighThresh, const num target, const num tolerance, const num kBond, const bool display) {
    state->changedBonds = true;
    state->redoNeighbors = true;

//	const int operationsPerStepMax = 10 ;
	const num neighThreshOrig = neighThresh;
    num neighThreshSqr = neighThresh * neighThresh;
	State *raw = state.get();
	bool is2d = raw->is2d;
	num maxIncrease = 1.1;
    vector<vector<num> > sigmas = state->atomParams->fillParams("sig");
    num maxRad = 0;
    for (uint i=0; i<sigmas.size(); i++) {
        sigmas[i] = LISTMAP(num, num, s, sigmas[i], s*neighThresh);
        maxRad = fmax(maxRad, *max_element(sigmas[i].begin(), sigmas[i].end()));
    }
	AtomGrid grid(state, maxIncrease * maxRad + .01, maxIncrease * maxRad + .01, 1);
	Bounds bounds = *raw->bounds;
	grid.deleteNeighbors();
	grid.populateLists();
	vector<Atom> &src = raw->atoms;
	vector<Atom *> atoms;
	atoms.reserve(src.size());
	for (Atom &a : src) {
		atoms.push_back(&a);//list map being super weird
	}
	vector<Bond> &allBonds = raw->bonds;

	vector<vector<Bond *> > bondss = Mod::musterBonds(state);

	vector<int> problemAtomIdxs;
	auto getZ = [&] () {
		return 2 * allBonds.size() / (num) atoms.size();
	};
	auto updateProblemAtoms = [&] () {
		problemAtomIdxs = Mod::atomsSingleSide(state, atoms, allBonds);
		return true;
	};
    vector<bool> haveBuilt;
    haveBuilt.reserve(atoms.size());
    for (uint i=0; i<atoms.size(); i++) {
        haveBuilt.push_back(false);
    }
	default_random_engine generator;
	random_device randDev;
	generator.seed(randDev());
	uniform_int_distribution<int> distribution(0, INT_MAX);
	num z = getZ();
	updateProblemAtoms();
	bool increasedSizeOnTurn = false;
	while ((abs(target - z) > tolerance or problemAtomIdxs.size())) {
		vector<Atom *> problemAtoms = LISTMAP(int, Atom *, i, problemAtomIdxs, atoms[i]);
		for (int i=0; i<(int) problemAtomIdxs.size(); i++) {
			int atomIdx = problemAtomIdxs[i];
			Atom *a = atoms[atomIdx];
			if (not haveBuilt[atomIdx]) {
				grid.buildNeighborlistRedund(a, maxRad);
                haveBuilt[atomIdx] = true;
			}
			vector<Neighbor> &neighbors = a->neighbors;
			vector<Bond *> bonds = bondss[atomIdx];
			vector<Atom *> bondedAtoms = LISTMAP(Bond *, Atom *, b, bonds, b->other(a));
            vector<Atom *> pickFrom;
            for (Neighbor neigh : neighbors) {
                num sigma = sigmas[a->type][neigh.obj->type];
                if (find(bondedAtoms.begin(), bondedAtoms.end(), neigh.obj) == bondedAtoms.end() and a->pos.distSqr(neigh.obj->pos + neigh.offset * bounds.trace) < neighThreshSqr*sigma*sigma) {
                    pickFrom.push_back(neigh.obj);
                }
            }
			if (!pickFrom.size() and !increasedSizeOnTurn) {
				neighThresh *= 1.03;	
                neighThreshSqr = neighThresh * neighThresh;
				increasedSizeOnTurn = true;
				if (neighThresh > maxIncrease * neighThreshOrig) {
					return false;
				}
			}
			if (pickFrom.size()) {
				int pickIdx = distribution(generator) % pickFrom.size();
				Atom *picked = pickFrom[pickIdx];
				auto found = find(problemAtoms.begin(), problemAtoms.end(), picked);

				if (found != problemAtoms.end()) {
					int eraseIdx = found - problemAtoms.begin();
					if (eraseIdx > i) {
						problemAtoms.erase(found);
						problemAtomIdxs.erase(problemAtomIdxs.begin() + eraseIdx); 
					}
				}
				Vector dist = a->pos.loopedVTo(picked->pos, bounds.trace);
                Bond *orig = &state->bonds[0];
				raw->addBond(a, picked, kBond, dist.len());
                if (orig != &state->bonds[0]) { //did I realloc?
                    bondss = Mod::musterBonds(state);
                } else {
                    Bond *last = &(*(raw->bonds.end()-1));
                    bondss[a-*atoms.begin()].push_back(last);
                    bondss[picked-*atoms.begin()].push_back(last);
                }
			}

		}
		
		z = getZ();
		int minValid = raw->is2d ? 3 : 4;
		//can check if removing the bond will just make more work later.  Should write a function to do this
		if (abs(z-target) > tolerance) {
			const int operationsPerStep = fabs(z - target) * atoms.size() / 2;
			if (z > target) {
				vector<int> potentialRemove;
				for (unsigned int bondsIdx=0; bondsIdx<bondss.size(); bondsIdx++) {
					if ((int) bondss[bondsIdx].size() > minValid) {//change this too
						potentialRemove.push_back(bondsIdx);
					}
				}
				assert(potentialRemove.size());
				int i=0;
                while (i < operationsPerStep ) {
                    int idxPick = distribution(generator) % potentialRemove.size();
                    if ((int) bondss[potentialRemove[idxPick]].size() > minValid) {
                        int targetIdxA = potentialRemove[idxPick];
                        vector<Bond *> &bonds = bondss[targetIdxA];			
                        Atom *targetAtom = atoms[targetIdxA];
                        int toPruneIdx = distribution(generator) % bonds.size();
                        Bond *b = bonds[toPruneIdx]; //could cause problems with cutting other atom's important bond, but let's add some randomness!
                        vector<Bond> bondsCopy = LISTMAPTEST(Bond *, Bond, testBond, bonds, *testBond, testBond != b);
                        if (not Mod::atomSingleSide(targetAtom, bondsCopy, is2d, bounds.trace)) {
                            Atom *other = b->other(targetAtom);
                            int otherIdx = other - &src[0];
                            vector<Bond *> &otherBonds = bondss[otherIdx];
                            vector<Bond> otherBondsCopy = LISTMAPTEST(Bond *, Bond, testBond, otherBonds, *testBond, testBond != b);
                            if (not Mod::atomSingleSide(other, otherBondsCopy, is2d, bounds.trace)) {
                                raw->removeBond(b);
                                i++;
                            }
                        } 

                    }

                    bondss = Mod::musterBonds(state);
                    z = getZ();
                }
			} else {
				int i=0;
				while (i < operationsPerStep) {
					int tryAdd = distribution(generator) % atoms.size();
					Atom *a = atoms[tryAdd];
					vector<Bond *> &bonds = bondss[tryAdd];
					vector<Atom *> bondedAtoms = LISTMAP(Bond *, Atom *, b, bonds, b->other(a));
                    if (not haveBuilt[tryAdd]) {
                        grid.buildNeighborlistRedund(a, maxRad);
                        haveBuilt[tryAdd] = true;
                    }
                    vector<Neighbor> &neighbors = a->neighbors;
                    vector<Atom *> pickFrom;
                    for (Neighbor neigh : neighbors) {
                        num sigma = sigmas[a->type][neigh.obj->type];
                        if (find(bondedAtoms.begin(), bondedAtoms.end(), neigh.obj) == bondedAtoms.end() and a->pos.distSqr(neigh.obj->pos + neigh.offset * bounds.trace) < neighThreshSqr * sigma*sigma) {
                            pickFrom.push_back(neigh.obj);
                        }
                    }
					if (pickFrom.size()) {
						int bondToIdx = distribution(generator) % pickFrom.size();
						Atom *picked = pickFrom[bondToIdx];
						Vector dist = a->pos.loopedVTo(picked->pos, bounds.trace);
                        Bond *orig = &state->bonds[0];
						raw->addBond(a, picked, kBond, dist.len());
                        if (orig != &state->bonds[0]) { //did I re-alloc?
                            bondss = Mod::musterBonds(state);
                        } else {
                            Bond *last = &(*(raw->bonds.end()-1));
                            bondss[a-*atoms.begin()].push_back(last);
                            bondss[picked-*atoms.begin()].push_back(last);
                        }
                        i++;
					}
					z = getZ();

				}
			}
		}
		updateProblemAtoms();
		
		increasedSizeOnTurn = false;
		if (display) {
            state->integrater->writeOutput();
		}
	}
	return true;
}



*/


__global__ void Mod::unskewAtoms(float4 *xs, int nAtoms, float3 xOrig, float3 yOrig, float3 lo) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        float lxo = length(xOrig);
        float lyo = length(yOrig);
        float lxf = xOrig.x;
        float lyf = yOrig.y;
       
        float a = atan2(xOrig.y, xOrig.x);
        float b = atan2(yOrig.x, yOrig.y);

        float invDenom = 1.0f / (lxo*lyo*cos(a)*cos(b) - lxo*lyo*sin(a)*sin(b));

        float c1 = lyo*cos(b) * invDenom;
        float c2 = -lyo*sin(b) * invDenom;
        float c3 = -lxo*sin(a) * invDenom;
        float c4 = lxo*cos(a) * invDenom;
        


        float4 pos = xs[idx];
        float xo = pos.x - lo.x;
        float yo = pos.y - lo.y;
        pos.x = lxf * (xo*c1 + yo*c2) + lo.x;
        pos.y = lyf * (xo*c3 + yo*c4) + lo.y;
        xs[idx] = pos;
    }
}


__global__ void Mod::skewAtomsFromZero(float4 *xs, int nAtoms, float3 xFinal, float3 yFinal, float3 lo) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        const double a = atan2(xFinal.y, xFinal.x);
        const double b = atan2(yFinal.x, yFinal.y);
        const double lxf = length(xFinal);
        const double lyf = length(yFinal);
        
        const double lxo = xFinal.x;
        const double lyo = yFinal.y;

        const double c1 = lxf*cos(a);
        const double c2 = lyf*sin(b);

        const double c3 = lxf*sin(a);
        const double c4 = lyf*cos(b);


        float4 pos = xs[idx];

        float xo = pos.x - lo.x;
        float yo = pos.y - lo.y;

        const double fx = xo / lxo;
        const double fy = yo / lyo;
        pos.x = fx * c1 + fy * c2 + lo.x;
        pos.y = fx * c3 + fy * c4 + lo.y;

        xs[idx] = pos;
    }
}

// CPU versions

void Mod::scaleAtomCoords(SHARED(State) state, string groupHandle, Vector around, Vector scaleBy) {
    return scaleAtomCoords(state.get(), groupHandle, around, scaleBy);
}
void Mod::scaleAtomCoords(State *state, string groupHandle, Vector around, Vector scaleBy) {
	//don't need to makeready for this one		
	//this should work for skewed bounds
	int groupTag = state->groupTagFromHandle(groupHandle);
	for (Atom &a : state->atoms) {
		if (a.groupTag & groupTag) {
			Vector diff = a.pos - around;
			diff *= scaleBy;
			a.pos = around + diff;
		}
	}

}

void Mod::unskewAtoms(vector<Atom> &atoms, Vector xOrig, Vector yOrig) {
    const double lxo = xOrig.len();
    const double lyo = yOrig.len();
    const double lxf = xOrig[0];
    const double lyf = yOrig[1];
    
    const double a = atan2(xOrig[1], xOrig[0]);
    const double b = atan2(yOrig[0], yOrig[1]);

    const double denom = lxo*lyo*cos(a)*cos(b) - lxo*lyo*sin(a)*sin(b);

    const double c1 = lyo*cos(b) / denom;
    const double c2 = -lyo*sin(b) / denom;
    const double c3 = -lxo*sin(a) / denom;
    const double c4 = lxo*cos(a) / denom;

	for (Atom &a : atoms) {
        double xo = a.pos[0];
        double yo = a.pos[1];
        a.pos[0] = lxf * (xo*c1 + yo*c2);
        a.pos[1] = lyf * (xo*c3 + yo*c4);
	}
}


void Mod::skewAtomsFromZero(vector<Atom> &atoms, Vector xFinal, Vector yFinal) {
    const double a = atan2(xFinal[1], xFinal[0]);
    const double b = atan2(yFinal[0], yFinal[1]);
    const double lxf = xFinal.len();
    const double lyf = yFinal.len();
    
    const double lxo = xFinal[0];
    const double lyo = yFinal[1];

    const double c1 = lxf*cos(a);
    const double c2 = lyf*sin(b);

    const double c3 = lxf*sin(a);
    const double c4 = lyf*cos(b);

	for (Atom &a : atoms) {
        const double fx = a.pos[0] / lxo;
        const double fy = a.pos[1] / lyo;
        a.pos[0] = fx * c1 + fy * c2;
        a.pos[1] = fx * c3 + fy * c4;
	}

}

void Mod::skewAtoms(vector<Atom> &atoms, Vector xOrig, Vector xFinal, Vector yOrig, Vector yFinal) {
    const double lxo = xOrig.len();
    const double lyo = yOrig.len();
    const double lxf = xFinal.len();
    const double lyf = yFinal.len();

    const double ao = atan2(xOrig[1], xOrig[0]);
    const double bo = atan2(yOrig[0], yOrig[1]);

    const double af = atan2(xFinal[1], xFinal[0]);
    const double bf = atan2(yFinal[0], yFinal[1]);
    //these coefficients are hairy enough that and functions in Vector just wouldn't be portable.  Going to write it here
    const double denom = (lxo* lyo* cos(ao)* cos(bo) - lxo* lyo*sin(ao)* sin(bo));
//four coefficients for x term
    const double c1 = lxf * lyo * cos(af) * cos(bo);
    const double c2 = lxo * lyf *  cos(ao) * sin(bf);
    const double c3 = lxo * lyf * sin(ao) * sin(bf);
    const double c4 = lxf * lyo *  cos(af)* sin(bo);

    const double c5 = lxo * lyf * cos(ao) * cos(bf);
    const double c6 = lxf * lyo * cos(bo) * sin(af);
    const double c7 = lxo * lyf * cos(bf) * sin(ao); 
    const double c8 = lxf * lyo * sin(af) * sin(bo);

    for (Atom &a : atoms) {
        double xo = a.pos[0];
        double yo = a.pos[1];
        a.pos[0] = (c1*xo + c2*yo - c3*xo - c4*yo) / denom;
        a.pos[1] = (c5*yo + c6*xo - c7*xo - c8*yo) / denom;
    }


//p[0] =(lxf * lyo * xo * cos(af) *  cos(bo) + lxo * lyf * yo * cos(ao) * sin(bf) - lxo * lyf * xo* sin(ao) * sin(bf) - lxf * lyo * yo * cos(af)* sin(bo))/(lxo* lyo* cos(ao)* cos(bo) - lxo* lyo*sin(ao)* sin(bo))

//p[1] = (lxo* lyf* yo* cos(ao)* cos(bf) + lxf* lyo* xo* cos(bo)* sin(af) -   lxo* lyf* xo* cos(bf)* sin(ao) -   lxf* lyo* yo* sin(af)* sin(bo))/(lxo* lyo* cos(ao)* cos(bo) - lxo* lyo*sin(ao)* sin(bo))

    

}


void Mod::skew(SHARED(State) state, Vector skewBy) { //x component is how much to shear y principle vector, y is for x vector
	State *raw = state.get();
	Bounds &b = raw->bounds;
    Vector xOrig = b.sides[0];
    Vector yOrig = b.sides[1];
    b.skew(skewBy);
    Vector xFinal = b.sides[0];
    Vector yFinal = b.sides[1];
	Mod::skewAtoms(raw->atoms, xOrig, xFinal, yOrig, yFinal);
}
