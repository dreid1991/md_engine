#include "DataManager.h"
#include "State.h"
DataManager::DataManager(State * state_) : state(state_) {
    //eng = DataSetAvg(state, "eng", 1, state->dataIntervalStd);
   // vol = DataSetVol(state, "vol", state->dataIntervalStd);
   // press = DataSetPress(state, "press", 1, state->dataIntervalStd);
   // sets.push_back((DataSet *) &eng);
   // sets.push_back((DataSet *) &vol);
   // sets.push_back((DataSet *) &press);

	
}
/*
//HEY, YOU'LL HAVE TO MANUALLY CHANGE accumulateEvery IF YOU'RE USING FUNKY TIMESTEPS ON FORCING THINGS
SHARED(DataSet) DataManager::createVol(string handle, int processEvery) {
	SHARED(DataSet) set ((DataSet *) new DataSetVol(state, handle, processEvery));
	setMap[handle] = set;
	sets.push_back(set);
	return set;
}

SHARED(DataSet) DataManager::createAvg(string handle, int accumulateEvery, int processEvery) {
	SHARED(DataSet) set ((DataSet *) new DataSetAvg(state, handle, accumulateEvery, processEvery));
	setMap[handle] = set;
	sets.push_back(set);
	return set;
}

SHARED(DataSet) DataManager::createPress(string handle, int accumulateEvery, int processEvery) {
	SHARED(DataSet) set ((DataSet *) new DataSetPress(state, handle, accumulateEvery, processEvery));
	setMap[handle] = set;
	sets.push_back(set);
	return set;
}
*/
SHARED(DataSet) DataManager::createPython(string handle, int processEvery, PyObject *py) {
	SHARED(DataSet) set ((DataSet *) new DataSetPython(state, handle, processEvery, py));
	userSets.push_back(set);
	return set;
}


SHARED(DataSet) DataManager::getDataSet(string handle) {
    for (SHARED(DataSet) d : userSets) {
        if (d->handle == handle) {
            return d;
        }
    }
    cout << "Failed to get data set with handle " << handle << endl;
    cout << "existing sets are " << endl;
    for (SHARED(DataSet) d : userSets) {
        cout << d->handle << endl;
    }
    assert(false);
    return SHARED(DataSet) ((DataSet *) NULL);
}


void export_DataManager() {
    class_<DataManager, SHARED(DataManager) >("DataManager", init<>())
        .def("createPython", &DataManager::createPython)
        .def("getDataSet", &DataManager::getDataSet)
        ;
}
