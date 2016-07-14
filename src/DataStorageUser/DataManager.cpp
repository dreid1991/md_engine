#include "DataManager.h"
#include "State.h"
#include "DataSetTemperature.h"
#include "DataSetEnergy.h"
#include "DataSetBounds.h"
using namespace std;
namespace py = boost::python;
DataManager::DataManager(State * state_) : state(state_) {
    turnLastEngs = state->turn-1;
}

DataManager::computeVirials() {
    if (turnLastEngs != state->turn) {
        state->integUtil.singlePointEng();
        turnLastEngs = state->turn;
    }
}

template <class T>
SHARED(T) recordGeneric(State *state, string groupHandle, vector<SHARED(T)> &dataSets, int collectEvery, py::object collectGenerator) {
    uint32_t groupTag = state->groupTagFromHandle(groupHandle);
    bool setExists = false;
    SHARED(T) dataSet;
    for (SHARED(T) ds : dataSets) {
        if (ds->sameGroup(groupTag)) {
            dataSet = ds;
            setExists = true;
        }
    }
    if (not setExists) {
        dataSet = SHARED(T) (new T(groupTag));
        dataSets.push_back(dataSet);
    }
    dataSet->takeCollectValues(collectEvery, collectGenerator);
    dataSet->setNextCollectTurn(state->turn);
    return dataSet;
}

template <class T>
void stopRecordGeneric(State *state, string dataType, string groupHandle, vector<SHARED(T)> &dataSets) {
    uint32_t groupTag = state->groupTagFromHandle(groupHandle);
    for (int i=0; i<dataSets.size(); i++) {
        SHARED(T) ds = dataSets[i];
        if (ds->sameGroup(groupTag)) {
            dataSets.erase(dataSets.begin()+i,  dataSets.begin()+i+1);
            return;
        }
    }
    cout << "Could not find data set to erase.  Type of data: " << dataType << " group handle " << groupHandle << endl;
    assert(false);

}

SHARED(DataSetTemperature) DataManager::recordTemperature(string groupHandle, int collectEvery, py::object collectGenerator) {
    return recordGeneric(state, groupHandle, dataSetsTemperature, collectEvery, collectGenerator);

}
void DataManager::stopRecordTemperature(string groupHandle) {
    stopRecordGeneric(state, "temperature", groupHandle, dataSetsTemperature);
}
SHARED(DataSetEnergy) DataManager::recordEnergy(string groupHandle, int collectEvery, py::object collectGenerator) {
    return recordGeneric(state, groupHandle, dataSetsEnergy, collectEvery, collectGenerator);

}
void DataManager::stopRecordEnergy(string groupHandle) {
    stopRecordGeneric(state, "energy", groupHandle, dataSetsEnergy);
}
SHARED(DataSetBounds) DataManager::recordBounds(int collectEvery, py::object collectGenerator) {
    return recordGeneric(state, "all", dataSetsBounds, collectEvery, collectGenerator);

}
void DataManager::stopRecordBounds() {
    stopRecordGeneric(state, "bounds", "all", dataSetsBounds);
}

void DataManager::generateSingleDataSetList() {
    //template this out or something
    dataSets = vector<DataSet *>();
    for (SHARED(DataSetTemperature) dst : dataSetsTemperature) {
        dataSets.push_back(dst.get());
    }
    for (SHARED(DataSetEnergy) dst : dataSetsEnergy) {
        dataSets.push_back(dst.get());
    }
    for (SHARED(DataSetBounds) dst : dataSetsBounds) {
        dataSets.push_back(dst.get());
    }
}
/*

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

bool DataManager::recordingEng() {
    return activeEngTags.size();
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

*/
void export_DataManager() {
    boost::python::class_<DataManager>(
        "DataManager",
        boost::python::no_init
    )
    .def("recordTemperature", &DataManager::recordTemperature,
            (boost::python::arg("handle") = "all",
             boost::python::arg("collectEvery") = 0,
             boost::python::arg("collectGenerator") = boost::python::object())
        )
    .def("stopRecordTemperature", &DataManager::stopRecordTemperature,
            (boost::python::arg("handle") = "all")
        )
    .def("recordEnergy", &DataManager::recordEnergy,
            (boost::python::arg("handle") = "all",
             boost::python::arg("collectEvery") = 0,
             boost::python::arg("collectGenerator") = boost::python::object())
        )
    .def("stopRecordEnergy", &DataManager::stopRecordEnergy,
            (boost::python::arg("handle") = "all")
        )
    .def("recordBounds", &DataManager::recordBounds,
            (boost::python::arg("collectEvery") = 0,
             boost::python::arg("collectGenerator") = boost::python::object())
        )
    .def("stopRecordBounds", &DataManager::stopRecordBounds)
 //   .def("getDataSet", &DataManager::getDataSet)
    ;
}
