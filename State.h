#ifndef STATE_H
#define STATE_H

#define RCUT_INIT -1
#define PADDING_INIT 0.5

#include <assert.h>
#include <iostream>

#include <map>
#include <tuple>
#include <vector>
#include <functional>
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
#include "DataSet.h"
#include "DataManager.h"

#include "boost_for_export.h"


void export_State();

class ReadConfig;
class Atom;
class Fix;
//class DataManager;
class WriteConfig;

// TODO: MAKE DESTRUCTOR THAT CALLS FINISH IF IT HASN'T BEEN CALLED
class State {
	bool removeGroupTag(string handle);
	uint addGroupTag(string handle);
	public:
		// Sooo GPU ones are active during runtime, 
		//		non-GPU are active during process (wording?) time.
		vector<Atom> atoms;
		GridGPU gridGPU;
		BoundsGPU boundsGPU;
		vector<Bond> bonds;
		// using tuples makes boost say invalid template parameter.  Not sure why.
		// Using int lists instead.  Also, can't have vector of static lists.
		// Don't want to use pair, b/c angle has 3 :(
		vector<int*> bondAtomIds; 
		GPUData gpd;
		AtomGrid grid;
		Bounds bounds;
		vector<Fix *> fixes;
		vector<SHARED(Fix)> fixesShr;
		DataManager data;
		vector<SHARED(WriteConfig) > writeConfigs;
		map<string, unsigned int> groupTags;
		bool is2d;
		bool buildNeighborlists;
		bool periodic[3];
		float dt;
		int turn;
		int runningFor;
		int runInit;
		int dangerousRebuilds;
		int periodicInterval;
		int dataIntervalStd;

		double rCut;
		double padding;
		
		bool activateFix(SHARED(Fix));
		bool deactivateFix(SHARED(Fix));
		bool activateWriteConfig(SHARED(WriteConfig));
		bool deactivateWriteConfig(SHARED(WriteConfig));
		//bool fixIsActive(SHARED(Fix));
		bool changedAtoms;
		bool changedBonds;
		bool changedGroups;
		bool redoNeighbors;
		bool addToGroupPy(string, boost::python::list);
		bool addToGroup(string, std::function<bool (Atom *)> );
		vector<Atom *> selectGroup(string handle);
		bool destroyGroup(string);
		bool createGroup(string, boost::python::list atoms=boost::python::list());
		uint groupTagFromHandle(string handle); 
		bool addAtom(string handle, Vector pos, double q);
		bool addAtomDirect(Atom);
		//bool addBond(Atom *, Atom *, double k, double rEq);
		void refreshBonds();
		void initData();
		bool removeAtom(Atom *);
		bool removeBond(Bond *);

		// because it's an unordered set, the elements will always be unique
		// use atom.id values, not Atom values, to allow for map/set hashing

		int addSpecies(string handle, double mass);

		int idxFromId(int id);
		Atom *atomFromId(int id);
		bool verbose;
		int shoutEvery;
		AtomParams atomParams;
		vector<Atom> copyAtoms();	
		vector<BondSave> copyBonds();
		void setAtoms(vector<Atom> &);
		void setBonds(vector<BondSave> &);
		void deleteBonds();  // SEANQ: what's the difference between remove and delete?
		void deleteAtoms();
		bool atomInGroup(Atom &, string handle);
		bool asyncHostOperation(std::function<void (int )> cb);
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
		// SEAN: maybe some typedefs could make this a little clearer
		void setNeighborSpecialsGeneric(
				std::function< vector<pair<int, vector<int> > > (Fix *)> processFix,
				std::function<void (vector<int> &, vector<int> &, int) > processEnd);
		void setNeighborlistExclusions();
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

	// Can't I just make the properties accessable rather than making get/set functions?
	// yes
	// SEAN: if you want to pave the road to hell

};

// SEANQ: is there a reason these are down below?
//#include "AtomGrid.h"

#endif

