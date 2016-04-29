#include "Fix.h"
#include "Bounds.h"
#include "AtomParams.h"
//#include "DataManager.h"
#include "WriteConfig.h"
#include "ReadConfig.h"
#include "PythonOperation.h"

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
	groupTags["all"] = (unsigned int) 1;
    //! \todo I think it would be great to also have the group "none"
    //!       in addition to "all"
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
	changedAtoms = true;
	changedGroups = true;
	shoutEvery = 5000;
	for (int i=0; i<3; i++) {
		periodic[i]=true;
	}
    //! \todo It would be nice to set verbose true/false in Logging.h and use
    //!       it for mdMessage.
	verbose = true;
    readConfig = SHARED(ReadConfig) (new ReadConfig(this));
    atomParams = AtomParams(this);
    computeVirials = false; //will be set to true if a fix need it (like barostat) during run setup
	dataManager = DataManager(this);
    specialNeighborCoefs[0] = 0;
    specialNeighborCoefs[1] = 0;
    specialNeighborCoefs[2] = 0.5;
    rng_is_seeded = false;
}


uint State::groupTagFromHandle(std::string handle) {
	assert(groupTags.find(handle) != groupTags.end());
	return groupTags[handle];
}

bool State::atomInGroup(Atom &a, std::string handle) {
    uint tag = groupTagFromHandle(handle);
    return a.groupTag & tag;
}

int State::addAtom(std::string handle, Vector pos, double q) {
	std::vector<std::string> &handles = atomParams.handles;
	auto it = find(handles.begin(), handles.end(), handle);
	assert(it != handles.end());
	int idx = it - handles.begin();//okay, so index in handles is type
	Atom a(pos, idx, -1, atomParams.masses[idx], q);
    bool added = addAtomDirect(a);
    if (added) {
        return atoms.back().id;
    } 
    return -1;
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
		std::cout << "Bad atom type " << a.type << std::endl;
		return false;
	}

	if (a.mass == -1 or a.mass == 0) {
		a.mass = atomParams.masses[a.type];
	}
	if (is2d) {
        if (fabs(a.pos[2]) > 0.2) { //some noise value if you aren't applying fix 2d every turn.  Should add override
            std::cout << "adding atom with large z value in 2d simulation. Not adding atom" << std::endl;
            return false;
        }
        a.pos[2] = 0;
	}

	atoms.push_back(a);
	changedAtoms = true;
	return true;
}
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
	int idx = a - &atoms[0];
	atoms.erase(atoms.begin()+idx, atoms.begin()+idx+1);	
	
	changedAtoms = true;

	redoNeighbors = true;

	return true;
}


int State::idxFromId(int id) {
    //! \todo Is variable ii really more efficient?
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


/*  use atomParams.addSpecies
int State::addSpecies(std::string handle, double mass) {
    int id = atomParams.addSpecies(handle, mass);
    if (id != -1) {
        for (Fix *f : fixes) {
            f->addSpecies(handle);
        }
    }
    return id;
}
*/
void State::setSpecialNeighborCoefs(float onetwo, float onethree, float onefour) {
    specialNeighborCoefs[0] = onetwo;
    specialNeighborCoefs[1] = onethree;
    specialNeighborCoefs[2] = onefour;
}

template <class T>
int getSharedIdx(std::vector<SHARED(T)> &list, SHARED(T) other) {
    for (unsigned int i=0; i<list.size(); i++) {
        if (list[i]->handle == other->handle) {
            return i;
        }
    }
    return -1;
}

template <class T>
bool removeGeneric(std::vector<SHARED(T)> &list, std::vector<T *> *unshared, SHARED(T) other) {
    int idx = getSharedIdx<T>(list, other);
    if (idx == -1) {
        return false;
    }
    list.erase(list.begin()+idx, list.begin()+idx+1);
    if (unshared != (std::vector<T *> *) NULL) {
        unshared->erase(unshared->begin()+idx, unshared->begin()+idx+1);
    }
    return true;
}

template <class T>
bool addGeneric(std::vector<SHARED(T)> &list, std::vector<T *> *unshared, SHARED(T) other) {
    int idx = getSharedIdx<T>(list, other);
    if (idx != -1) {
        return false;
    }
    bool added = false;
    for (int idx=0; idx<list.size(); idx++) {
        SHARED(T)  existing = list[idx];
        if (other->orderPreference < existing->orderPreference) {
            list.insert(list.begin() + idx, other);
            if (unshared != (std::vector<T *> *) NULL) {
                unshared->insert(unshared->begin() + idx, other.get());
            }
            added = true;
            break;

        }

    }
    if (not added) {
        list.insert(list.end(), other);
        if (unshared != (std::vector<T *> *) NULL) {
            unshared->insert(unshared->end(), other.get());
        }

    }
    return true;
}

bool State::activateWriteConfig(SHARED(WriteConfig) other) {
    return addGeneric<WriteConfig>(writeConfigs, (std::vector<WriteConfig *> *) NULL, other);
}
bool State::deactivateWriteConfig(SHARED(WriteConfig) other) {
    return removeGeneric<WriteConfig>(writeConfigs, (std::vector<WriteConfig *> *) NULL, other);
}

bool State::activatePythonOperation(SHARED(PythonOperation) other) {
    return addGeneric<PythonOperation>(pythonOperations, (std::vector<PythonOperation *> *) NULL, other);
}
bool State::deactivatePythonOperation(SHARED(PythonOperation) other) {
    return removeGeneric<PythonOperation>(pythonOperations, (std::vector<PythonOperation *> *) NULL, other);
}


bool State::activateFix(SHARED(Fix) other) {
    if (other->state != this) {
        std::cout << "Trying to add fix with handle " << other->handle
                  << ", but fix was initialized with a different State" << std::endl;
    }
    assert(other->state == this);
    return addGeneric<Fix>(fixesShr, &fixes, other);
}
bool State::deactivateFix(SHARED(Fix) other) {
    return removeGeneric<Fix>(fixesShr, &fixes, other);
}

float State::getMaxRCut() {
    float maxRCut = 0;
    for (Fix *f : fixes) {
        vector<float> rCuts = f->getRCuts();
        for (float x : rCuts) {
            maxRCut = fmax(x, maxRCut);
        }
    }
    return maxRCut;
}
bool State::prepareForRun() {
    //fixes have already prepared by the time the integrater calls this prepare
    int nAtoms = atoms.size();
    std::vector<float4> xs_vec, vs_vec, fs_vec, fsLast_vec;
    std::vector<uint> ids;
    std::vector<float> qs;
    xs_vec.reserve(nAtoms);
    vs_vec.reserve(nAtoms);
    fs_vec.reserve(nAtoms);
    fsLast_vec.reserve(nAtoms);
    ids.reserve(nAtoms);
    qs.reserve(nAtoms);

    for (Atom &a : atoms) {
        xs_vec.push_back(make_float4(a.pos[0], a.pos[1], a.pos[2], 
                         *(float *)&a.type));
        vs_vec.push_back(make_float4(a.vel[0], a.vel[1], a.vel[2], 1/a.mass));
        fs_vec.push_back(make_float4(a.force[0], a.force[1], a.force[2], 
                         *(float *)&a.groupTag));
        fsLast_vec.push_back(
                make_float4(a.forceLast[0], a.forceLast[1], a.forceLast[2], 0));
        ids.push_back(a.id);
        qs.push_back(a.q);
    }
    gpd.xs.set(xs_vec);
    gpd.vs.set(vs_vec); 

    gpd.fs.set(fs_vec);

    gpd.fsLast.set(fsLast_vec);
    gpd.ids.set(ids);
    gpd.qs.set(qs);
    std::vector<int> id_vec = LISTMAPREF(Atom, int, a, atoms, a.id);
    std::vector<int> idToIdxs_vec;
    int size = *max_element(id_vec.begin(), id_vec.end()) + 1;
    //so... wanna keep ids tightly packed.  That's managed by program, not user
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
    float maxRCut = getMaxRCut();
    gridGPU = grid.makeGPU(maxRCut);
    gpd.xsBuffer = GPUArrayGlobal<float4>(nAtoms);
    gpd.vsBuffer = GPUArrayGlobal<float4>(nAtoms);
    gpd.fsBuffer = GPUArrayGlobal<float4>(nAtoms);
    gpd.fsLastBuffer = GPUArrayGlobal<float4>(nAtoms);
    gpd.idsBuffer = GPUArrayGlobal<uint>(nAtoms);
    gpd.perParticleEng = GPUArrayGlobal<float>(nAtoms);

    return true;
}

void copyAsyncWithInstruc(State *state, std::function<void (int64_t )> cb, int64_t turn) {
    cudaStream_t stream;
    CUCHECK(cudaStreamCreate(&stream));
    state->gpd.xsBuffer.dataToHostAsync(stream);
    state->gpd.vsBuffer.dataToHostAsync(stream);
    state->gpd.fsBuffer.dataToHostAsync(stream);
    state->gpd.fsLastBuffer.dataToHostAsync(stream);
    state->gpd.idsBuffer.dataToHostAsync(stream);
    CUCHECK(cudaStreamSynchronize(stream));
    std::vector<int> idToIdxsOnCopy = state->gpd.idToIdxsOnCopy;
    std::vector<float4> &xs = state->gpd.xsBuffer.h_data;
    std::vector<float4> &vs = state->gpd.vsBuffer.h_data;
    std::vector<float4> &fs = state->gpd.fsBuffer.h_data;
    std::vector<float4> &fsLast = state->gpd.fsLastBuffer.h_data;
    std::vector<uint> &ids = state->gpd.idsBuffer.h_data;
    std::vector<Atom> &atoms = state->atoms;

    for (int i=0, ii=state->atoms.size(); i<ii; i++) {
        int id = ids[i];
        int idxWriteTo = idToIdxsOnCopy[id];
        atoms[idxWriteTo].pos = xs[i];
        atoms[idxWriteTo].vel = vs[i];
        atoms[idxWriteTo].force = fs[i];
        atoms[idxWriteTo].forceLast = fsLast[i];
    }
    cb(turn);
    CUCHECK(cudaStreamDestroy(stream));
}

bool State::asyncHostOperation(std::function<void (int64_t )> cb) {
    // buffers should already be allocated in prepareForRun, and num atoms
    // shouldn't have changed.
    gpd.xs.copyToDeviceArray((void *) gpd.xsBuffer.getDevData());
    gpd.vs.copyToDeviceArray((void *) gpd.vsBuffer.getDevData());
    gpd.fs.copyToDeviceArray((void *) gpd.fsBuffer.getDevData());
    gpd.fsLast.copyToDeviceArray((void *) gpd.fsLastBuffer.getDevData());
    gpd.ids.copyToDeviceArray((void *) gpd.idsBuffer.getDevData());
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
    std::vector<float4> &xs = gpd.xs.h_data;
    std::vector<float4> &vs = gpd.vs.h_data;
    std::vector<float4> &fs = gpd.fs.h_data;
    std::vector<float4> &fsLast = gpd.fsLast.h_data;
    std::vector<uint> &ids = gpd.ids.h_data;
    for (int i=0, ii=atoms.size(); i<ii; i++) {
        int id = ids[i];
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
	//	refreshBonds();//this must go before doing pbc, otherwise atom pointers could be messed up when you get atom offsets for bonds, which happens in pbc
	}
	if ((changedAtoms || redoNeighbors)) {
        //grid->periodicBoundaryConditions(); //UNCOMMENT THIS, WAS DONE FOR CUDA INITIAL STUFF
	}
	
	changedAtoms = false;
	changedGroups = false;
	redoNeighbors = false;
	return true;
}

bool State::addToGroupPy(std::string handle, boost::python::list toAdd) {//testF takes index, returns bool
	int tagBit = groupTagFromHandle(handle);  //if I remove asserts from this, could return things other than true, like if handle already exists
    int len = boost::python::len(toAdd);
    for (int i=0; i<len; i++) {
        boost::python::extract<Atom *> atomPy(toAdd[i]);
        if (!atomPy.check()) {
            cout << "Invalid atom found when trying to add to group" << endl;
            assert(atomPy.check());
        }
        Atom *a = atomPy;
        if (not (a >= &atoms[0] and a <= &atoms.back())) {
            std::cout << "Tried to add atom that is not in the atoms list.  "
                      << "If you added or removed atoms after taking a "
                      << "reference to this atom, the list storing atoms may "
                      << "have moved in memory, making this an invalid pointer."
                      << "  Consider resetting your atom variables"
                      << std::endl;
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

bool State::addToGroup(std::string handle, std::function<bool (Atom *)> testF) {
	int tagBit = addGroupTag(handle);
	for (Atom &a : atoms) {
		if (testF(&a)) {
			a.groupTag |= tagBit;
		}
	}
	changedGroups = true;
	return true;
}

bool State::destroyGroup(std::string handle) {
	uint tagBit = groupTagFromHandle(handle);
	assert(handle != "all");
	for (Atom &a : atoms) {
		a.groupTag &= ~tagBit;
	}
	removeGroupTag(handle);
	changedGroups = true;
	return true;
}

bool State::createGroup(std::string handle, boost::python::list forGrp) {
    uint res = addGroupTag(handle);
    if (!res) {
        std::cout << "Tried to create group " << handle
                  << " << that already exists" << std::endl;
        return false;
    }
    if (boost::python::len(forGrp)) {
        addToGroupPy(handle, forGrp);
    }
    return true;
}

uint State::addGroupTag(std::string handle) {
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

bool State::removeGroupTag(std::string handle) {
	auto it = groupTags.find(handle);
	assert(it != groupTags.end());
	groupTags.erase(it);
	return true;
}

std::vector<Atom *> State::selectGroup(std::string handle) {
	int tagBit = groupTagFromHandle(handle);
	return LISTMAPREFTEST(Atom, Atom *, a, atoms, &a, a.groupTag & tagBit);
}

std::vector<Atom> State::copyAtoms() {
	std::vector<Atom> save;
	save.reserve(atoms.size());
	for (Atom &a : atoms) {
		Atom copy = a;
		copy.neighbors = vector<Neighbor>();
		save.push_back(copy);
	}
	return save;
}



bool State::validAtom(Atom *a) {
    return a >= atoms.data() and a <= &atoms.back();
}

void State::deleteAtoms() {
	atoms.erase(atoms.begin(), atoms.end());
}

void State::setAtoms(std::vector<Atom> &fromSave) {
	changedAtoms = true;
	changedGroups = true;
	atoms = fromSave;
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
    deleteAtoms();
}

std::mt19937 &State::getRNG() {
    if (!rng_is_seeded) {
        seedRNG();
    }
    return randomNumberGenerator;
}

void State::seedRNG(unsigned int seed) {
    if (seed == 0) {
        random_device randDev;
        randomNumberGenerator.seed(randDev());
    } else {
        randomNumberGenerator.seed(seed);
    }
    rng_is_seeded = true;
}

    //helper for reader funcs (LAMMPS reader)
Vector generateVector(State &s) {
    return Vector();
}

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(State_seedRNG_overloads,State::seedRNG,0,1)

void export_State() {
    boost::python::class_<State,
                          SHARED(State) >(
        "State",
        boost::python::init<>()
    )
    .def("addAtom", &State::addAtom,
            (boost::python::arg("handle"),
             boost::python::arg("pos"),
             boost::python::arg("q")=0)
        )
    .def_readonly("atoms", &State::atoms)
    .def("setPeriodic", &State::setPeriodic)
    .def("getPeriodic", &State::getPeriodic) //boost is grumpy about readwriting static arrays.  can readonly, but that's weird to only allow one w/ wrapper func for other.  doing wrapper funcs for both
    .def("removeAtom", &State::removeAtom)
    //.def("removeBond", &State::removeBond)
    .def("idxFromId", &State::idxFromId)

    .def("addToGroup", &State::addToGroupPy)
    .def("destroyGroup", &State::destroyGroup)
    .def("createGroup", &State::createGroup,
            (boost::python::arg("handle"),
             boost::python::arg("atoms") = boost::python::list())
        )
    .def("selectGroup", &State::selectGroup)
    .def("copyAtoms", &State::copyAtoms)
    .def("setAtoms", &State::setAtoms)

    .def("setSpecialNeighborCoefs", &State::setSpecialNeighborCoefs)

    .def("activateFix", &State::activateFix)
    .def("deactivateFix", &State::deactivateFix)
    .def("activateWriteConfig", &State::activateWriteConfig)
    .def("deactivateWriteConfig", &State::deactivateWriteConfig)
    .def("activatePythonOperation", &State::activatePythonOperation)
    .def("deactivatePythonOperation", &State::deactivatePythonOperation)
    .def("zeroVelocities", &State::zeroVelocities)
    .def("destroy", &State::destroy)
    .def("seedRNG", &State::seedRNG, State_seedRNG_overloads())
    .def_readwrite("is2d", &State::is2d)
    .def_readonly("changedAtoms", &State::changedAtoms)
    .def_readonly("changedGroups", &State::changedGroups)
    .def_readwrite("buildNeighborlists", &State::buildNeighborlists)
    .def_readwrite("turn", &State::turn)
    .def_readwrite("periodicInterval", &State::periodicInterval)
    .def_readwrite("rCut", &State::rCut)
    .def_readwrite("dt", &State::dt)    
    .def_readwrite("padding", &State::padding)
    .def_readonly("groupTags", &State::groupTags)
    .def_readonly("dataManager", &State::dataManager)
    //shared ptrs
    .def_readwrite("grid", &State::grid)
    .def_readwrite("bounds", &State::bounds)
    .def_readwrite("fixes", &State::fixes)
    .def_readwrite("atomParams", &State::atomParams)
    .def_readwrite("writeConfigs", &State::writeConfigs)
    .def_readonly("readConfig", &State::readConfig)
    .def_readwrite("shoutEvery", &State::shoutEvery)
    .def_readwrite("verbose", &State::verbose)
    .def_readonly("deviceManager", &State::devManager)
    //helper for reader funcs
    .def("Vector", &generateVector)


    ;

}


