#pragma once
#include "DataSet.h"
#include "globalDefs.h"
#include <boost/shared_ptr.hpp>
#include "boost_for_export.h"
class State;
void export_DataManager();
//okay - energy and pressure ptrs ARE in aux, but they will always be the zeroth and first entries, respectively.  They are just seperated for easy access. 
//
//

class DataPoint {
    public:
        int timestep;
        double value;
        DataPoint(double timestep_, double value_) : timestep(timestep_), value(value_) {
        }
};
class DataManager {
	public:
		State *state;
        int dataInterval;
		DataManager(){};
		DataManager(State *); //ugh, state will always be around while data manager is active.  want it internally created, no access to shared ptr that way
        void collectData();
        bool recordEng(string groupHandle);
        bool stopRecordEng(string groupHandle);
        bool recordingEng();
        vector<string> activeEngHandles;
        vector<uint> activeEngTags;
        map<string, vector<DataPoint> > engData;         
        vector<SHARED(DataSet) > userSets;
        SHARED(DataSet) createPython(string handle, int computeEvery, PyObject *py);
        SHARED(DataSet) getDataSet(string handle);
};

/*
 *
 have global data collection frequency

 okay... so I can say state.dataManager.recordEng(groupHandle)
 state.dataManager.recordEngPerParticle(groupHandle)

 *
 *
 *
 *
 *
 */
