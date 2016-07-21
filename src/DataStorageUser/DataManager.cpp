#include "DataManager.h"
#include "State.h"
#include "DataComputer.h"
#include "DataComputerTemperature.h"
#include "DataSetUser.h"
using namespace MD_ENGINE;
namespace py = boost::python;
DataManager::DataManager(State * state_) : state(state_) {
    turnLastEngs = state->turn-1;
}

//okay - assumption: energies are computed rarely.  I can get away with not computing them in force kernels and just computing them when a data set needs them
void DataManager::computeEnergy() {
    if (turnLastEngs != state->turn) {
        state->integUtil.singlePointEng();
        turnLastEngs = state->turn;
    }
}


/*template <class T>
SHARED(T) recordGeneric(State *state, string groupHandle, vector<SHARED(T)> &dataSets, int interval, py::object collectGenerator) {
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
    dataSet->takeCollectValues(interval, collectGenerator);
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
*/

boost::shared_ptr<DataSetUser> DataManager::createDataSet(boost::shared_ptr<DataComputer> comp, uint32_t groupTag, int dataMode, int dataType, int interval, py::object collectGenerator) {
    if (interval == 0) {
        return boost::shared_ptr<DataSetUser>(new DataSetUser(state, comp, groupTag, dataMode, dataType, collectGenerator));
    } else {
        return boost::shared_ptr<DataSetUser>(new DataSetUser(state, comp, groupTag, dataMode, dataType, interval));
    }

}

boost::shared_ptr<DataSetUser> DataManager::recordTemperature(std::string groupHandle, int interval, py::object collectGenerator) { //add tensor, etc, later
    int dataType = DATATYPE::TEMPERATURE;
    boost::shared_ptr<DataComputer> comp = boost::shared_ptr<DataComputer> ( (DataComputer *) new DataComputerTemperature(state, true, false) );
    uint32_t groupTag = state->groupTagFromHandle(groupHandle);
    boost::shared_ptr<DataSetUser> dataSet = createDataSet(comp, groupTag, DATAMODE::SCALAR, DATATYPE::TEMPERATURE, interval, collectGenerator);
    dataSets.push_back(dataSet);
    return dataSet;

}
void DataManager::stopRecord(boost::shared_ptr<DataSetUser> dataSet) {
    for (int i=0; i<dataSets.size(); i++) {
        boost::shared_ptr<DataSetUser> ds = dataSets[i];
        if (ds == dataSet) {
            dataSets.erase(dataSets.begin()+i);
            break;
        }
    }
}
/*
SHARED(DataSetEnergy) DataManager::recordEnergy(string groupHandle, int interval, py::object collectGenerator) {
    return recordGeneric(state, groupHandle, dataSetsEnergy, interval, collectGenerator);

}
}
SHARED(DataSetBounds) DataManager::recordBounds(int interval, py::object collectGenerator) {
    return recordGeneric(state, "all", dataSetsBounds, interval, collectGenerator);

}
*/
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
    py::class_<DataManager>(
        "DataManager",
        py::no_init
    )
    .def("recordTemperature", &DataManager::recordTemperature,
            (py::arg("handle") = "all",
             py::arg("interval") = 0,
             py::arg("turnGenerator") = py::object())
        )
    /*
    .def("stopRecordTemperature", &DataManager::stopRecordTemperature,
            (py::arg("handle") = "all")
        )
    .def("recordEnergy", &DataManager::recordEnergy,
            (py::arg("handle") = "all",
             py::arg("interval") = 0,
             py::arg("collectGenerator") = py::object())
        )
    .def("stopRecordEnergy", &DataManager::stopRecordEnergy,
            (py::arg("handle") = "all")
        )
    .def("recordBounds", &DataManager::recordBounds,
            (py::arg("interval") = 0,
             py::arg("collectGenerator") = py::object())
        )
    .def("stopRecordBounds", &DataManager::stopRecordBounds)
    */
 //   .def("getDataSet", &DataManager::getDataSet)
    ;
}
