#include "DataManager.h"
#include "State.h"
#include "DataComputer.h"
#include "DataComputerTemperature.h"
#include "DataComputerEnergy.h"
#include "DataComputerPressure.h"
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




boost::shared_ptr<DataSetUser> DataManager::createDataSet(boost::shared_ptr<DataComputer> comp, uint32_t groupTag, int dataMode, int dataType, int interval, py::object collectGenerator) {
    if (interval == 0) {
        return boost::shared_ptr<DataSetUser>(new DataSetUser(state, comp, groupTag, dataMode, dataType, collectGenerator));
    } else {
        return boost::shared_ptr<DataSetUser>(new DataSetUser(state, comp, groupTag, dataMode, dataType, interval));
    }

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


boost::shared_ptr<DataSetUser> DataManager::recordTemperature(std::string groupHandle, int interval, py::object collectGenerator) { //add tensor, etc, later
    boost::shared_ptr<DataComputer> comp = boost::shared_ptr<DataComputer> ( (DataComputer *) new DataComputerTemperature(state, true, false) );
    uint32_t groupTag = state->groupTagFromHandle(groupHandle);
    boost::shared_ptr<DataSetUser> dataSet = createDataSet(comp, groupTag, DATAMODE::SCALAR, DATATYPE::TEMPERATURE, interval, collectGenerator);
    dataSets.push_back(dataSet);
    return dataSet;

}

boost::shared_ptr<DataSetUser> DataManager::recordEnergy(std::string groupHandle, int interval, py::object collectGenerator) {
    int dataType = DATATYPE::ENERGY;
    boost::shared_ptr<DataComputer> comp = boost::shared_ptr<DataComputer> ( (DataComputer *) new DataComputerEnergy(state) );
    uint32_t groupTag = state->groupTagFromHandle(groupHandle);
    boost::shared_ptr<DataSetUser> dataSet = createDataSet(comp, groupTag, DATAMODE::SCALAR, DATATYPE::ENERGY, interval, collectGenerator);
    dataSets.push_back(dataSet);
    return dataSet;


}

boost::shared_ptr<DataSetUser> DataManager::recordPressure(std::string groupHandle, int interval, py::object collectGenerator) {
    int dataType = DATATYPE::PRESSURE;
    boost::shared_ptr<DataComputer> comp = boost::shared_ptr<DataComputer> ( (DataComputer *) new DataComputerPressure(state, true, false) );
    uint32_t groupTag = state->groupTagFromHandle(groupHandle);
    //deal with tensors later
    boost::shared_ptr<DataSetUser> dataSet = createDataSet(comp, groupTag, DATAMODE::SCALAR, DATATYPE::PRESSURE, interval, collectGenerator);
    dataSets.push_back(dataSet);
    return dataSet;


}

void DataManager::addVirialTurn(int64_t t) {
    virialTurns.insert(t);
}
void DataManager::clearVirialTurn(int64_t turn) {
    auto it = virialTurns.find(turn);
    if (it != virialTurns.end()) {
        virialTurns.erase(it);
    }
}
/*

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
    .def("stopRecord", &DataManager::stopRecord)

    .def("recordTemperature", &DataManager::recordTemperature,
            (py::arg("handle") = "all",
             py::arg("interval") = 0,
             py::arg("turnGenerator") = py::object())
        )
    .def("recordEnergy", &DataManager::recordEnergy,
            (py::arg("handle") = "all",
             py::arg("interval") = 0,
             py::arg("collectGenerator") = py::object())
        )
    .def("recordPressure", &DataManager::recordPressure,
            (py::arg("handle") = "all",
             py::arg("interval") = 0,
             py::arg("collectGenerator") = py::object())
        )
   /* 
    .def("recordBounds", &DataManager::recordBounds,
            (py::arg("interval") = 0,
             py::arg("collectGenerator") = py::object())
        )
    .def("stopRecordBounds", &DataManager::stopRecordBounds)
    */
 //   .def("getDataSet", &DataManager::getDataSet)
    ;
}
