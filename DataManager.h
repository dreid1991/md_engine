#ifndef DATAMANAGER_H
#define DATAMANAGER_H
#include "DataSet.h"
#include "globalDefs.h"
#include <boost/shared_ptr.hpp>
#include "boost_for_export.h"
class State;
void export_DataManager();
//okay - energy and pressure ptrs ARE in aux, but they will always be the zeroth and first entries, respectively.  They are just seperated for easy access. 
class DataManager {
	public:
		State *state;
		DataManager(){};
		DataManager(State *); //ugh, state will always be around while data manager is active.  want it internally created, no access to shared ptr that way
        vector<SHARED(DataSet) > userSets;
        SHARED(DataSet) createPython(string handle, int computeEvery, PyObject *py);
        SHARED(DataSet) getDataSet(string handle);
};

#endif
