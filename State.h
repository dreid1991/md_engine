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
	bool removeGroupTag(std::string handle);
	uint addGroupTag(std::string handle);
    float getMaxRCut();//<!to be called from within prepare for run __after__ fixed have prepared (because that removes any DEFAULT_FILL values)
	public:
		// Sooo GPU ones are active during runtime, 
		//		non-GPU are active during process (wording?) time.
		std::vector<Atom> atoms;
		GridGPU gridGPU;
		BoundsGPU boundsGPU;
		GPUData gpd;
        DeviceManager devManager;
		AtomGrid grid;
		Bounds bounds;
		std::vector<Fix *> fixes;
		std::vector<SHARED(Fix)> fixesShr;
		DataManager dataManager;
		std::vector<SHARED(WriteConfig) > writeConfigs;
        std::vector<SHARED(PythonOperation) > pythonOperations;
		std::map<std::string, uint32_t> groupTags;
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


		double rCut; //!< rCut behavior - each pair fix can define its own rCuts.  If values are not defined, this value is used as the default.  When a run begins, the grid will figure out the largest rCut out of the pairs and use that value.  This bit has not been implemented yet.
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
		bool addToGroupPy(std::string, boost::python::list);
		bool addToGroup(std::string, std::function<bool (Atom *)> );
		std::vector<Atom *> selectGroup(std::string handle);
		bool destroyGroup(std::string);
		bool createGroup(std::string, boost::python::list atoms=boost::python::list());
		uint32_t groupTagFromHandle(std::string handle);
		int addAtom(std::string handle, Vector pos, double q);
		bool addAtomDirect(Atom);
		bool removeAtom(Atom *);

		// because it's an unordered set, the elements will always be unique
		// use atom.id values, not Atom values, to allow for map/set hashing

		int addSpecies(std::string handle, double mass);

		int idxFromId(int id);
		Atom *atomFromId(int id);
		bool verbose;
		int shoutEvery;
		AtomParams atomParams;
		std::vector<Atom> copyAtoms();
		void setAtoms(std::vector<Atom> &);
		void deleteAtoms();
		bool atomInGroup(Atom &, std::string handle);
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
		std::vector<int> idxFromIdCache;
		void updateIdxFromIdCache();
		
		int maxIdExisting;
		std::vector<int> idBuffer;
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

