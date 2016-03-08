#include "DataManager.h"
#include "State.h"
DataManager::DataManager(State * state_) : state(state_) {

	
}


SHARED(DataSet) DataManager::createPython(string handle, int processEvery, PyObject *py) {
	SHARED(DataSet) set ((DataSet *) new DataSetPython(state, handle, processEvery, py));
	userSets.push_back(set);
	return set;
}

bool DataManager::recordEng(string groupHandle) {
    uint groupTag = state->groupTagFromHandle(groupHandle); //will assert false if handle doesn't exist
    if (find(activeEngTags.begin(), activeEngTags.end(), groupTag) == activeEngTags.end()) {
        activeEngTags.push_back(groupTag);
        activeEngHandles.push_back(groupHandle);
        if (engData.find(groupHandle) == engData.end()) {
            engData[groupHandle] = vector<DataPoint>();
        }

        return true;
    }
    return false;
}
bool DataManager::stopRecordEng(string groupHandle) {
    uint groupTag = state->groupTagFromHandle(groupHandle); //will assert false if handle doesn't exist
    auto it = find(activeEngTags.begin(), activeEngTags.end(), groupTag);
    if (it != activeEngTags.end()) {
        int idx = it - activeEngTags.begin();
        activeEngTags.erase(activeEngTags.begin()+idx, activeEngTags.begin()+idx+1);
        activeEngHandles.erase(activeEngHandles.begin()+idx, activeEngHandles.begin()+idx+1);
        return true;
    }
    return false;
}
//called by integrater
void DataManager::collectData() {
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
