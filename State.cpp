
/* State is where everything is sewn together. We set global options:
 *   - gpu cuda device data and options
 *   - atoms and groups
 *   - spatial options like grid/bound options, periodicity
 *   - fixes (local and shared)
 *   - neighbor lists
 *   - timestep
 *   - turn
 *   - cuts
 * */

#include "State.h"

State::State() {
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	groupTags["all"] = (unsigned int) 1;
	is2d = false;
	buildNeighborlists = true;
	rCut = RCUT_INIT;
	padding = PADDING_INIT;
	turn = 0;
    maxIdExisting = -1;
    maxExclusions = 0;
	dangerousRebuilds = 0;
	dt = .005;
	periodicInterval = 50;
	dataIntervalStd = 50;
	changedAtoms = true;
	changedGroups = true;
	changedBonds = true;
	shoutEvery = 5000;
	for (int i=0; i<3; i++) {
		periodic[i]=true;
	}
	verbose = true;
	initData();
    readConfig = SHARED(ReadConfig) (new ReadConfig(this));
    data = DataManager(this);
    atomParams = AtomParams(this);
}

void State::initData() {
	//data = SHARED(DataManager) (new DataManager(this));
}

uint State::groupTagFromHandle(string handle) {
	assert(groupTags.find(handle) != groupTags.end());
	return groupTags[handle];
}

bool State::atomInGroup(Atom &a, string handle) {
    uint tag = groupTagFromHandle(handle);
    return a.groupTag & tag;
}

bool State::addAtom(string handle, Vector pos, double q) {
	vector<string> &handles = atomParams.handles;
	auto it = find(handles.begin(), handles.end(), handle);
	assert(it != handles.end());
	int idx = it - handles.begin();//okay, so index in handles is type
	Atom a(pos, idx, -1, atomParams.masses[idx], q);
	return addAtomDirect(a);
}

bool State::addAtomDirect(Atom a) {
    //id of atom a WILL be overridden
    if (idBuffer.size()) {
        a.id = idBuffer.back();
        idBuffer.pop_back();
    } else {
        maxIdExisting++;
        a.id = maxIdExisting;
    }
    
	if (a.type >= atomParams.numTypes) {
		cout << "Bad atom type " << a.type << endl;
		return false;
	}

	if (a.mass == -1 or a.mass == 0) {
		a.mass = atomParams.masses[a.type];
	}
	if (is2d) {
        if (fabs(a.pos[2]) > 0.2) { //some noise value if you aren't applying fix 2d every turn.  Should add override
            cout << "adding atom with large z value in 2d simulation. Not adding atom" << endl;
            return false;
        }
        a.pos[2] = 0;
	}

	atoms.push_back(a);
	changedAtoms = true;
	return true;
}

//constructor should be same
/*
bool State::addBond(Atom *a, Atom *b, num k, num rEq) {
	if (a == b || 
        !(a >= &(*atoms.begin()) && a < &(*atoms.end())) || 
        !(b >= &(*atoms.begin()) && b < &(*atoms.end()))) {
		return false;
	}
	int *ids = (int *) malloc(sizeof(int) * 2);
	ids[0] = a->id;
	ids[1] = b->id;
	bondAtomIds.push_back(ids);
	Bond bond(a, b, k, rEq);
	bonds.push_back(bond);
	changedBonds = true;
	return true;
}
*/

bool State::removeAtom(Atom *a) {
	if (!(a >= &(*atoms.begin()) && a < &(*atoms.end()))) {
		return false;
	}
    int id = a->id;
    if (id == maxIdExisting) {
        maxIdExisting--;
        //need to collapse maxIdExisting to first space (of more that n1) in ids in idBuffer
        //this relies on sort line below
        while (idBuffer.size() and maxIdExisting == idBuffer.back()) {
            idBuffer.pop_back();
            maxIdExisting--;
        }
    } else {
        idBuffer.push_back(id);
        sort(idBuffer.begin(), idBuffer.end());
    }

	bool erasedBonds = false;
	for (int i=bonds.size()-1; i>=0; i--) {
		Bond &b = bonds[i];
		if (b.hasAtom(a)) {
			bonds.erase(bonds.begin()+i, bonds.begin()+i+1);
			free(bondAtomIds[i]);
			bondAtomIds.erase(bondAtomIds.begin()+i, bondAtomIds.begin()+i+1);
			erasedBonds = true;
		}
	}

	int idx = a - &atoms[0];
	atoms.erase(atoms.begin()+idx, atoms.begin()+idx+1);	
	
	changedAtoms = true;
	if (erasedBonds) {
		changedBonds = true; 
	}
	redoNeighbors = true;

	return true;
}

bool State::removeBond(Bond *b) {
	if (!(b >= &(*bonds.begin()) && b < &(*bonds.end()))) {
		return false;
	}
	int idx = b - &(*bonds.begin());
	bonds.erase(bonds.begin()+idx, bonds.begin()+idx+1);
	free(bondAtomIds[idx]);
	bondAtomIds.erase(bondAtomIds.begin()+idx, bondAtomIds.begin()+idx+1);
	changedBonds = true;
	return true;
}

int State::idxFromId(int id) {
	for (int i=0,ii=atoms.size(); i<ii; i++) {
		if (atoms[i].id == id) {
			return i;
		}
	}
	return -1;
}
void State::updateIdxFromIdCache() {
    idxFromIdCache = vector<int>(maxIdExisting+1);
    for (int i=0; i<atoms.size(); i++) {
        idxFromIdCache[atoms[i].id] = i;
    }
}
Atom *State::atomFromId(int id) {
	for (int i=0,ii=atoms.size(); i<ii; i++) {
		if (atoms[i].id == id) {
			return &atoms[i];
		}
	}
	return NULL; 
}

void State::refreshBonds() {
	assert(bondAtomIds.size() == bonds.size());
	vector<int> ids = LISTMAPREF(Atom, int, a, atoms, a.id);
	for (int i=0, ii=bonds.size(); i<ii; i++) {
		int idA = bondAtomIds[i][0];
		int idB = bondAtomIds[i][1];
		int idxA = find(ids.begin(), ids.end(), idA) - ids.begin();
		int idxB = find(ids.begin(), ids.end(), idB) - ids.begin();
		assert(idxA != idxB);
		bonds[i].atoms[0] = &atoms[idxA];
		bonds[i].atoms[1] = &atoms[idxB];

	}
}

int State::addSpecies(string handle, double mass) {
    int id = atomParams.addSpecies(handle, mass);
    if (id != -1) {
        for (Fix *f : fixes) {
            f->addSpecies(handle);
        }
    }
    return id;
}

template <class T>
int getSharedIdx(vector<SHARED(T)> &list, SHARED(T) other) {
    for (unsigned int i=0; i<list.size(); i++) {
        if (list[i]->handle == other->handle) {
            return i;
        }
    }
    return -1;
}

template <class T>
bool removeGeneric(vector<SHARED(T)> &list, vector<T *> *unshared, SHARED(T) other) {
    int idx = getSharedIdx<T>(list, other);
    if (idx == -1) {
        return false;
    }
    list.erase(list.begin()+idx, list.begin()+idx+1);
    if (unshared != (vector<T *> *) NULL) {
        unshared->erase(unshared->begin()+idx, unshared->begin()+idx+1);
    }
    return true;
}

template <class T>
bool addGeneric(vector<SHARED(T)> &list, vector<T *> *unshared, SHARED(T) other) {
    int idx = getSharedIdx<T>(list, other);
    if (idx != -1) {
        return false;
    }
    bool added = false;
    for (int idx=0; idx<list.size(); idx++) {
        SHARED(T)  existing = list[idx];
        if (other->orderPreference < existing->orderPreference) {
            list.insert(list.begin() + idx, other);
            if (unshared != (vector<T *> *) NULL) {
                unshared->insert(unshared->begin() + idx, other.get());
            }
            added = true;
            break;

        }

    }
    if (not added) {
        list.insert(list.end(), other);
        if (unshared != (vector<T *> *) NULL) {
            unshared->insert(unshared->end(), other.get());
        }

    }
    return true;
}

bool State::deactivateWriteConfig(SHARED(WriteConfig) other) {
    return removeGeneric<WriteConfig>(writeConfigs, (vector<WriteConfig *> *) NULL, other);

}
bool State::activateWriteConfig(SHARED(WriteConfig) other) {
    return addGeneric<WriteConfig>(writeConfigs, (vector<WriteConfig *> *) NULL, other);
}


bool State::activateFix(SHARED(Fix) other) {
    return addGeneric<Fix>(fixesShr, &fixes, other);
}
bool State::deactivateFix(SHARED(Fix) other) {
    return removeGeneric<Fix>(fixesShr, &fixes, other);
}

void State::setNeighborSpecialsGeneric(
        std::function< vector<pair<int, vector<int> > > (Fix *)> processFix, 
        std::function<void (vector<int> &, vector<int> &, int) > processEnd) {
    vector<pair<int, set<int> > > exclusions;
    int maxPerAtom = 0;
    int nExcl = 0;
    for (Fix *f : fixes) {
        vector<pair<int, vector<int> > > fixExclusions = processFix(f);
        // outer one has to be a vector b/c set elements can't be modified 
        //f->neighborlistExclusions(); 
        // each fix MUST return a sorted list (by id) from neighborlistExclusions
        int exclIdx = 0;
        for (pair<int, vector<int> > &atomExclusions : fixExclusions) {
            int atomId = atomExclusions.first;
            while (exclIdx < exclusions.size() and atomId > exclusions[exclIdx].first) {
                exclIdx++;
            }
            if (exclIdx == exclusions.size()) {
                pair<int, set<int> > toAdd;
                toAdd.first = atomId;
                toAdd.second.insert(atomExclusions.second.begin(), 
                                    atomExclusions.second.end());
                exclusions.push_back(toAdd);
            } else if (atomId == exclusions[exclIdx].first) {
                exclusions[exclIdx].second.insert(atomExclusions.second.begin(), 
                                                  atomExclusions.second.end());
            } else {
                //COULD BE SLOW
                pair<int, set<int> > toAdd;
                toAdd.first = atomId;
                toAdd.second.insert(atomExclusions.second.begin(), 
                                    atomExclusions.second.end());
                exclusions.insert(exclusions.begin() + exclIdx, toAdd);
            }
        }
    }
    
    //exclusions are sorted by atom id by construction

    //okay.  now we have them mustered.  Now let's convert to condensed list on gpu
    /*
    cout << "final exclusions" << endl;
    for (pair<int, set<int> > &excl : exclusions) {
        cout << "atom id " << excl.first << endl;
        for (int x : excl.second) {
            cout << x << " ";
        }
        cout << endl;
    }
    */
    {
        int maxId = maxIdExisting;
        vector<int> idxs;
        vector<int> excls;
        idxs.reserve(maxId + 1); 
        idxs.push_back(0);

        int workingId = 0;
        for (pair<int, set<int> > &atomExclusions : exclusions) {
            int atomId = atomExclusions.first;
            while (workingId < atomId) {
                idxs.push_back(idxs.back()); //filling in blanks
                workingId++;
            }
            maxPerAtom = fmax(maxPerAtom, atomExclusions.second.size());
            excls.insert(excls.end(), atomExclusions.second.begin(), 
                                      atomExclusions.second.end());
            idxs.push_back(idxs.back() + atomExclusions.second.size());
            workingId++;
        }
        processEnd(idxs, excls, maxPerAtom);
        /*
        gpd.nlistExclusions.set(excls);
        gpd.nlistExclusionIdxs.set(idxs);

        gpd.nlistExclusions.dataToDevice();
        gpd.nlistExclusionIdxs.dataToDevice();
        */

    }
}

void State::setNeighborlistExclusions() {
    /*
    maxExclusions = 0;
    auto processFix = [this] (Fix *f) {
        return f->neighborlistExclusions();
    };
    auto processEnd = [this] (vector<int> &idxs, vector<int> &excls, int maxPerAtom) {
        maxExclusions = maxPerAtom;
        gpd.nlistExclusions.set(excls);
        gpd.nlistExclusionIdxs.set(idxs);

        gpd.nlistExclusions.dataToDevice();
        gpd.nlistExclusionIdxs.dataToDevice();
    };
    setNeighborSpecialsGeneric(processFix, processEnd);
    vector<pair<int, set<int> > > exclusions;
    int maxPerAtom = 0;
    int nExcl = 0;
    for (Fix *f : fixes) {
        vector<pair<int, vector<int> > > fixExclusions = processFix(f);
        // outer one has to be a vector b/c set elements can't be modified 
        //f->neighborlistExclusions(); 
        //i each fix MUST return a sorted list (by id) from neighborlistExclusions
        int exclIdx = 0;
        for (pair<int, vector<int> > &atomExclusions : fixExclusions) {
            int atomId = atomExclusions.first;
            while (exclIdx < exclusions.size() and atomId > exclusions[exclIdx].first) {
                exclIdx++;
            }
            if (exclIdx == exclusions.size()) {
                pair<int, set<int> > toAdd;
                toAdd.first = atomId;
                toAdd.second.insert(atomExclusions.second.begin(), 
                                    atomExclusions.second.end());
                exclusions.push_back(toAdd);
            } else if (atomId == exclusions[exclIdx].first) {
                exclusions[exclIdx].second.insert(atomExclusions.second.begin(), 
                                                  atomExclusions.second.end());
            } else {
                //COULD BE SLOW
                pair<int, set<int> > toAdd;
                toAdd.first = atomId;
                toAdd.second.insert(atomExclusions.second.begin(), 
                                    atomExclusions.second.end());
                exclusions.insert(exclusions.begin() + exclIdx, toAdd);
            }
        }
    }
    
    //exclusions are sorted by atom id by construction
    //okay.  now we have them mustered.  Now let's convert to condensed list on gpu
    {
        int maxId = maxIdExisting;
        vector<int> idxs;
        vector<int> excls;
        idxs.reserve(maxId + 1); 
        idxs.push_back(0);

        int workingId = 0;
        for (pair<int, set<int> > &atomExclusions : exclusions) {
            int atomId = atomExclusions.first;
            while (workingId < atomId) {
                idxs.push_back(idxs.back()); //filling in blanks
                workingId++;
            }
            maxPerAtom = fmax(maxPerAtom, atomExclusions.second.size());
            excls.insert(excls.end(), atomExclusions.second.begin(), 
                                      atomExclusions.second.end());
            idxs.push_back(idxs.back() + atomExclusions.second.size());

            workingId++;
        }
        processEnd(idxs, excls);
    }
    */
}

bool State::prepareForRun() {
    int nAtoms = atoms.size();
    vector<float4> xs_vec, vs_vec, fs_vec, fsLast_vec;
    vector<short> types;
    vector<float> qs;
    xs_vec.reserve(nAtoms);
    vs_vec.reserve(nAtoms);
    fs_vec.reserve(nAtoms);
    fsLast_vec.reserve(nAtoms);
    types.reserve(nAtoms);
    qs.reserve(nAtoms);

    for (Atom &a : atoms) {
        xs_vec.push_back(make_float4(a.pos[0], a.pos[1], a.pos[2], 
                         *(float *)&a.id));
        vs_vec.push_back(make_float4(a.vel[0], a.vel[1], a.vel[2], 1/a.mass));
        fs_vec.push_back(make_float4(a.force[0], a.force[1], a.force[2], 
                         *(float *)&a.groupTag));
        fsLast_vec.push_back(
                make_float4(a.forceLast[0], a.forceLast[1], a.forceLast[2], 0));
        types.push_back(a.type);
        qs.push_back(a.q);
    }
    gpd.xs.set(xs_vec);
    gpd.vs.set(vs_vec); 

    gpd.fs.set(fs_vec);

    gpd.fsLast.set(fsLast_vec);
    gpd.types.set(types);
    gpd.qs.set(qs);
    vector<int> id_vec = LISTMAPREF(Atom, int, a, atoms, a.id);
    vector<int> idToIdxs_vec;
    int size = *max_element(id_vec.begin(), id_vec.end()) + 1;
    //so... wanna keep ids tightly packed.  That'll be managed by program, not user
    idToIdxs_vec.reserve(size);
    for (int i=0; i<size; i++) {
        idToIdxs_vec.push_back(-1);
    }
    for (int i=0; i<id_vec.size(); i++) {
        idToIdxs_vec[id_vec[i]] = i; 
    }
    gpd.idToIdxsOnCopy = idToIdxs_vec;
    gpd.idToIdxs.set(idToIdxs_vec);
    boundsGPU = bounds.makeGPU();
    gridGPU = grid.makeGPU();
    gpd.xsBuffer = GPUArray<float4>(nAtoms);
    gpd.vsBuffer = GPUArray<float4>(nAtoms);
    gpd.fsBuffer = GPUArray<float4>(nAtoms);
    gpd.fsLastBuffer = GPUArray<float4>(nAtoms);

    //setNeighborlistExclusions();
    return true;
}

void copyAsyncWithInstruc(State *state, std::function<void (int )> cb, int turn) {
    cudaStream_t stream;
    CUCHECK(cudaStreamCreate(&stream));
    state->gpd.xsBuffer.dataToHostAsync(stream);
    state->gpd.vsBuffer.dataToHostAsync(stream);
    state->gpd.fsBuffer.dataToHostAsync(stream);
    state->gpd.fsLastBuffer.dataToHostAsync(stream);
    CUCHECK(cudaStreamSynchronize(stream));
    vector<int> idToIdxsOnCopy = state->gpd.idToIdxsOnCopy;
    vector<float4> &xs = state->gpd.xsBuffer.h_data;
    vector<float4> &vs = state->gpd.vsBuffer.h_data;
    vector<float4> &fs = state->gpd.fsBuffer.h_data;
    vector<float4> &fsLast = state->gpd.fsLastBuffer.h_data;
    vector<Atom> &atoms = state->atoms;
    for (int i=0, ii=state->atoms.size(); i<ii; i++) {
        int id = *(int *) &xs[i].w;
        int idxWriteTo = idToIdxsOnCopy[id];
        atoms[idxWriteTo].pos = xs[i];
        atoms[idxWriteTo].vel = vs[i];
        atoms[idxWriteTo].force = fs[i];
        atoms[idxWriteTo].forceLast = fsLast[i];
    }
    cb(turn);
    CUCHECK(cudaStreamDestroy(stream));
}

bool State::asyncHostOperation(std::function<void (int )> cb) {
    // buffers should already be allocated in prepareForRun, and num atoms
    // shouldn't have changed.
    gpd.xs.copyToDeviceArray((void *) gpd.xsBuffer.getDevData());
    gpd.vs.copyToDeviceArray((void *) gpd.vsBuffer.getDevData());
    gpd.fs.copyToDeviceArray((void *) gpd.fsBuffer.getDevData());
    gpd.fsLast.copyToDeviceArray((void *) gpd.fsLastBuffer.getDevData());
    bounds.set(boundsGPU);
    if (asyncData and asyncData->joinable()) {
        asyncData->join();
    }
    cudaDeviceSynchronize();
    //cout << "ASYNC IS NOT ASYNC" << endl;
    //copyAsyncWithInstruc(this, cb, turn);
    asyncData = SHARED(thread) ( new thread(copyAsyncWithInstruc, this, cb, turn));
    // okay, now launch a thread to start async copying, then wait for it to
    // finish, and set the cb on state (and maybe a flag if you can't set the
    // function lambda equal to null
    //ONE MIGHT ASK WHY I'M NOT JUST DOING THE CALLBACK FROM THE THREAD
    //THE ANSWER IS BECAUSE I WANT TO USE THIS FOR PYTHON BIZ, AND YOU CAN'T
    //CALL PYTHON FUNCTIONS FROM A THREAD AS FAR AS I KNOW
    //if thread exists, wait for it to finish
    //okay, now I have all of these buffers filled with data.  now let's just
    //launch a thread which does the writing.  At the end of each just (in main
    //iteration code, just have a join statement before any of the downloading
    //happens
}

bool State::downloadFromRun() {
    vector<float4> &xs = gpd.xs.h_data;
    vector<float4> &vs = gpd.vs.h_data;
    vector<float4> &fs = gpd.fs.h_data;
    vector<float4> &fsLast = gpd.fsLast.h_data;
    for (int i=0, ii=atoms.size(); i<ii; i++) {
        int id = *(int *) &xs[i].w;
        int idxWriteTo = gpd.idToIdxsOnCopy[id];
        atoms[idxWriteTo].pos = xs[i];
        atoms[idxWriteTo].vel = vs[i];
        atoms[idxWriteTo].force = fs[i];
        atoms[idxWriteTo].forceLast = fsLast[i];
    }
    return true;
}

bool State::makeReady() {
	if (changedAtoms or changedGroups) {
		for (Fix* fix : fixes) {
			fix->refreshAtoms(); 
		}
	}
	if (changedAtoms) {
		refreshBonds();//this must go before doing pbc, otherwise atom pointers could be messed up when you get atom offsets for bonds, which happens in pbc
	}
	if ((changedAtoms || redoNeighbors)) {
        //grid->periodicBoundaryConditions(); //UNCOMMENT THIS, WAS DONE FOR CUDA INITIAL STUFF
	}
	
	changedAtoms = false;
	changedGroups = false;
	redoNeighbors = false;
	return true;
}

bool State::addToGroupPy(string handle, boost::python::list toAdd) {//testF takes index, returns bool
	int tagBit = groupTagFromHandle(handle);  //if I remove asserts from this, could return things other than true, like if handle already exists
    int len = boost::python::len(toAdd);
    for (int i=0; i<len; i++) {
        Atom *a = boost::python::extract<Atom *>(toAdd[i]);
        if (not (a >= &atoms[0] and a <= &atoms.back())) {
            cout << "Tried to add atom that is not in the atoms list.  If you added or removed atoms after taking a reference to this atom, the list storing atoms may have moved in memory, making this an invalid pointer.  Consider resetting your atom variables" << endl;
            assert(false);
        }
        a->groupTag |= tagBit;
    }
    /*
	for (unsigned int i=0; i<atoms.size(); i++) {
		PyObject *res = PyObject_CallFunction(testF, (char *) "i", i);
		assert(PyBool_Check(res));
		if (PyObject_IsTrue(res)) {
			atoms[i].groupTag |= tagBit;	
		} 
	}
    */
	return true;

}

bool State::addToGroup(string handle, std::function<bool (Atom *)> testF) {
	int tagBit = addGroupTag(handle);
	for (Atom &a : atoms) {
		if (testF(&a)) {
			a.groupTag |= tagBit;
		}
	}
	changedGroups = true;
	return true;
}

bool State::destroyGroup(string handle) {
	uint tagBit = groupTagFromHandle(handle);
	assert(handle != "all");
	for (Atom &a : atoms) {
		a.groupTag &= ~tagBit;
	}
	removeGroupTag(handle);
	changedGroups = true;
	return true;
}

bool State::createGroup(string handle, boost::python::list forGrp) {
    uint res = addGroupTag(handle);
    if (!res) {
        cout << "Tried to create group " << handle << " << that already exists" << endl;
        return false;
    }
    if (boost::python::len(forGrp)) {
        addToGroupPy(handle, forGrp);
    }
    return true;
}

uint State::addGroupTag(string handle) {
	uint working = 0;
	assert(groupTags.find(handle) == groupTags.end());
	for (auto it=groupTags.begin(); it!=groupTags.end(); it++) {
		working |= it->second;
	}
	for (int i=0; i<32; i++) {
		uint potentialTag = 1 << i;
		if (! (working & potentialTag)) {
			groupTags[handle] = potentialTag;
			return potentialTag;
		}
	}
	return 0;
}

bool State::removeGroupTag(string handle) {
	auto it = groupTags.find(handle);
	assert(it != groupTags.end());
	groupTags.erase(it);
	return true;
}

vector<Atom *> State::selectGroup(string handle) {
	int tagBit = groupTagFromHandle(handle);
	return LISTMAPREFTEST(Atom, Atom *, a, atoms, &a, a.groupTag & tagBit);
}

vector<Atom> State::copyAtoms() {
	vector<Atom> save;
	save.reserve(atoms.size());
	for (Atom &a : atoms) {
		Atom copy = a;
		copy.neighbors = vector<Neighbor>();
		save.push_back(copy);
	}
	return save;
}

vector<BondSave> State::copyBonds() {
    /*
	vector<BondSave> save;
	save.reserve(bonds.size());
	for (unsigned int i=0; i<bonds.size(); i++) {
		save.push_back(BondSave(bondAtomIds[i], bonds[i].k, bonds[i].rEq));
	}
	return save;
    */
}

void State::setBonds(vector<BondSave> &saved) {
    /*
	deleteBonds();
	for (BondSave &bond : saved) {
		assert(bond.ids[0] < (int) atoms.size() and bond.ids[1] < (int) atoms.size());
		Atom *a = &atoms[bond.ids[0]];
		Atom *b = &atoms[bond.ids[1]];
		num k = bond.k;
		num rEq = bond.rEq;
		addBond(a, b, k, rEq);	
	}
    */
}

bool State::validAtom(Atom *a) {
    return a >= atoms.data() and a <= &atoms.back();
}

void State::deleteAtoms() {
	atoms.erase(atoms.begin(), atoms.end());
}

void State::deleteBonds() {
    /*
	bonds = vector<Bond>();
	for (unsigned int i=0; i<bondAtomIds.size(); i++) {
		free(bondAtomIds[i]);
	}
	bondAtomIds = vector<int *>();
    */
}

void State::setAtoms(vector<Atom> &fromSave) {
	changedAtoms = true;
	changedGroups = true;
	changedBonds = true;
	atoms = fromSave;
	deleteBonds();
}


void State::zeroVelocities() {
	for (Atom &a : atoms) {
		a.vel.zero();
	}
}


void State::destroy() {
    //if (bounds) {  //UNCOMMENT
    //  bounds->state = NULL;
    //}
    //UNCOMMENT
    //bounds = NULL;
    deleteBonds();
    deleteAtoms();
}

void export_State() {
    class_<State, SHARED(State) >("State", init<>())
        .def("addAtom", &State::addAtom, (python::arg("handle"), python::arg("pos"), python::arg("q")=0) )
        .def_readonly("atoms", &State::atoms)
        .def("setPeriodic", &State::setPeriodic)
        .def("getPeriodic", &State::getPeriodic) //boost is grumpy about readwriting static arrays.  can readonly, but that's weird to only allow one w/ wrapper func for other.  doing wrapper funcs for both
        .def("removeAtom", &State::removeAtom)
        //.def("removeBond", &State::removeBond)
        .def("idxFromId", &State::idxFromId)

        .def("addToGroup", &State::addToGroupPy)
        .def("destroyGroup", &State::destroyGroup)
        .def("createGroup", &State::createGroup, (python::arg("handle"), python::arg("atoms") = boost::python::list()))
        .def("selectGroup", &State::selectGroup)
        .def("copyAtoms", &State::copyAtoms)
        .def("setAtoms", &State::setAtoms)

        .def("activateFix", &State::activateFix)
        .def("deactivateFix", &State::deactivateFix)
        .def("activateWriteConfig", &State::activateWriteConfig)
        .def("deactivateWriteConfig", &State::deactivateWriteConfig)
        .def("zeroVelocities", &State::zeroVelocities)
        .def("destroy", &State::destroy)
        .def_readwrite("dataIntervalStd", &State::dataIntervalStd)	
        .def_readwrite("is2d", &State::is2d)
        .def_readonly("changedAtoms", &State::changedAtoms)
        .def_readonly("changedBonds", &State::changedBonds)
        .def_readonly("changedGroups", &State::changedGroups)
        .def_readwrite("buildNeighborlists", &State::buildNeighborlists)
        .def_readwrite("turn", &State::turn)
        .def_readwrite("periodicInterval", &State::periodicInterval)
        .def_readwrite("rCut", &State::rCut)
        .def_readwrite("padding", &State::padding)
        .def_readonly("groupTags", &State::groupTags)
        .def_readonly("data", &State::data)
        //shared ptrs
        .def_readwrite("grid", &State::grid)
        .def_readwrite("bounds", &State::bounds)
        .def_readwrite("fixes", &State::fixes)
        .def_readwrite("atomParams", &State::atomParams)
        .def_readwrite("writeConfigs", &State::writeConfigs)
        .def_readonly("readConfig", &State::readConfig)
        .def_readwrite("shoutEvery", &State::shoutEvery)
        .def_readwrite("verbose", &State::verbose)

        ;

}


