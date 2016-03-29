#ifndef STATE_H
#define STATE_H

#define RCUT_INIT -1
#define PADDING_INIT 0.5

#include <assert.h>
#include <iostream>
#include <stdint.h>

#include <map>
#include <tuple>
#include <vector>
#include <functional>
#include <random>
#include <thread>

#include <boost/shared_ptr.hpp>
#include <boost/variant/get.hpp>
#include <boost/python.hpp>

#include "globalDefs.h"
#include "GPUArrayTex.h"
#include "GPUArray.h"

using namespace std;
using namespace boost;
using namespace boost::python;

class State;  //forward declaring so bond can use bounds, which includes state

#include "AtomParams.h"
#include "Atom.h"
#include "Bond.h"
#include "GPUData.h"
#include "GridGPU.h"
#include "Bounds.h"
#include "AtomGrid.h"
#include "DataManager.h"

#include "boost_for_export.h"
#include "DeviceManager.h"

void export_State();
class PythonOperation;
class ReadConfig;
class Fix;
//class DataManager;
class WriteConfig;

class State {
	bool removeGroupTag(string handle);
	uint addGroupTag(string handle);
	public:
		// Sooo GPU ones are active during runtime, 
		//		non-GPU are active during process (wording?) time.
		vector<Atom> atoms;
		GridGPU gridGPU;
		BoundsGPU boundsGPU;
		GPUData gpd;
        DeviceManager devManager;
		AtomGrid grid;
		Bounds bounds;
		vector<Fix *> fixes;
		vector<SHARED(Fix)> fixesShr;
		DataManager dataManager;
		vector<SHARED(WriteConfig) > writeConfigs;
        vector<SHARED(PythonOperation) > pythonOperations;
		map<string, uint32_t> groupTags;
		bool is2d;
		bool buildNeighborlists;
		bool periodic[3];
		float dt;
        float specialNeighborCoefs[3]; //as 1-2, 1-3, 1-4 neighbors
		int64_t turn;
		int runningFor;
		int64_t runInit;
		int dangerousRebuilds;
		int periodicInterval;

        bool computeVirials;


		double rCut;
		double padding;
		
        void setSpecialNeighborCoefs(float onetwo, float onethree, float onefour); 
		bool activateFix(SHARED(Fix));
		bool deactivateFix(SHARED(Fix));
		bool activateWriteConfig(SHARED(WriteConfig));
		bool deactivateWriteConfig(SHARED(WriteConfig));

        bool activatePythonOperation(SHARED(PythonOperation));
        bool deactivatePythonOperation(SHARED(PythonOperation));
		//bool fixIsActive(SHARED(Fix));
		bool changedAtoms;
		bool changedGroups;
		bool redoNeighbors;
		bool addToGroupPy(string, boost::python::list);
		bool addToGroup(string, std::function<bool (Atom *)> );
		vector<Atom *> selectGroup(string handle);
		bool destroyGroup(string);
		bool createGroup(string, boost::python::list atoms=boost::python::list());
		uint32_t groupTagFromHandle(string handle); 
		bool addAtom(string handle, Vector pos, double q);
		bool addAtomDirect(Atom);
		bool removeAtom(Atom *);

		// because it's an unordered set, the elements will always be unique
		// use atom.id values, not Atom values, to allow for map/set hashing

		int addSpecies(string handle, double mass);

		int idxFromId(int id);
		Atom *atomFromId(int id);
		bool verbose;
		int shoutEvery;
		AtomParams atomParams;
		vector<Atom> copyAtoms();	
		void setAtoms(vector<Atom> &);
		void deleteAtoms();
		bool atomInGroup(Atom &, string handle);
		bool asyncHostOperation(std::function<void (int64_t )> cb);
		SHARED(thread) asyncData;
		SHARED(ReadConfig) readConfig;

		State();
		
		void setPeriodic(int idx, bool val) {
			assert(idx > 0 and idx < 3);
			periodic[idx] = val;
		}
		bool getPeriodic(int idx) {
			assert(idx > 0 and idx < 3);
			return periodic[idx];
		}
        bool validAtom(Atom *);
		bool makeReady();
		int maxExclusions;
		bool prepareForRun();
		bool downloadFromRun();
		void zeroVelocities();
		void destroy();
		// these two are for managing atom ids such that they are densely packed
		// and it's quick at add atoms in large systems
		vector<int> idxFromIdCache; 
		void updateIdxFromIdCache();
		
		int maxIdExisting;
		vector<int> idBuffer;
		// Akay, so we declare grid, fixes, bounds, and integrator seperately

		std::mt19937 &getRNG();
		void seedRNG(unsigned int seed = 0);

	private:
		std::mt19937 randomNumberGenerator;
		bool rng_is_seeded;

	// Can't I just make the properties accessable rather than making get/set functions?
	// yes
	// SEAN: if you want to pave the road to hell

};

// SEANQ: is there a reason these are down below?
//#include "AtomGrid.h"

#endif

