#include "DataManager.h"
#include "State.h"
#include "DataComputer.h"
#include "DataComputerTemperature.h"
#include "DataComputerEnergy.h"
#include "DataComputerPressure.h"
#include "DataComputerBounds.h"
#include "DataComputerCOMV.h"
#include "DataComputerDipolarCoupling.h"
#include "DataComputerEField.h"
#include "DataComputerRDF.h"
#include "DataComputerHamiltonian.h"
#include "DataSetUser.h"
using namespace MD_ENGINE;
using std::set;
using std::pair;

namespace py = boost::python;

DataManager::DataManager(State * state_) : state(state_) {

}


boost::shared_ptr<DataSetUser> DataManager::createDataSet(boost::shared_ptr<DataComputer> comp, uint32_t groupTag, int interval, py::object collectGenerator) {
    if (interval == 0) {
        return boost::shared_ptr<DataSetUser>(new DataSetUser(state, comp, groupTag, collectGenerator));
    } else {
        return boost::shared_ptr<DataSetUser>(new DataSetUser(state, comp, groupTag, interval));
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


boost::shared_ptr<DataSetUser> DataManager::recordTemperature(std::string groupHandle, std::string computeMode, int interval, py::object collectGenerator) { //add tensor, etc, later
    boost::shared_ptr<DataComputer> comp = boost::shared_ptr<DataComputer> ( (DataComputer *) new DataComputerTemperature(state, computeMode) );
    uint32_t groupTag = state->groupTagFromHandle(groupHandle);
    boost::shared_ptr<DataSetUser> dataSet = createDataSet(comp, groupTag, interval, collectGenerator);
    dataSets.push_back(dataSet);
    return dataSet;
}


boost::shared_ptr<DataSetUser> DataManager::recordEnergy(std::string groupHandle, std::string computeMode, int interval, py::object collectGenerator, py::list fixes, std::string groupHandleB) {
    boost::shared_ptr<DataComputer> comp = boost::shared_ptr<DataComputer> ( (DataComputer *) new DataComputerEnergy(state, fixes, computeMode, groupHandleB) );
    uint32_t groupTag = state->groupTagFromHandle(groupHandle);
    boost::shared_ptr<DataSetUser> dataSet = createDataSet(comp, groupTag, interval, collectGenerator);
    dataSets.push_back(dataSet);
    return dataSet;
}

boost::shared_ptr<DataSetUser> DataManager::recordRDF(std::string species1, std::string species2, double binWidth, int interval, py::object collectGenerator) {
    boost::shared_ptr<DataComputer> comp = boost::shared_ptr<DataComputer> ( (DataComputer *) new DataComputerRDF(state, "vector", species1, species2, binWidth));
    uint32_t groupTag = state->groupTagFromHandle("all");
    boost::shared_ptr<DataSetUser> dataSet = createDataSet(comp, groupTag, interval, collectGenerator);
    dataSets.push_back(dataSet);
    return dataSet;
}

boost::shared_ptr<DataSetUser> DataManager::recordHamiltonian(std::string computeMode, int interval, py::object collectGenerator) { 
    std::string groupHandle = "all";
    boost::shared_ptr<DataComputer> comp = boost::shared_ptr<DataComputer> ( (DataComputer *) new DataComputerHamiltonian(state, computeMode) );
    uint32_t groupTag = state->groupTagFromHandle(groupHandle);
    boost::shared_ptr<DataSetUser> dataSet = createDataSet(comp, groupTag, interval, collectGenerator);
    dataSets.push_back(dataSet);
    return dataSet;
}

boost::shared_ptr<DataSetUser> DataManager::recordPressure(std::string groupHandle, std::string computeMode, int interval, py::object collectGenerator) {
    boost::shared_ptr<DataComputer> comp = boost::shared_ptr<DataComputer> ( (DataComputer *) new DataComputerPressure(state, computeMode) );
    uint32_t groupTag = state->groupTagFromHandle(groupHandle);
    //deal with tensors later
    boost::shared_ptr<DataSetUser> dataSet = createDataSet(comp, groupTag, interval, collectGenerator);
    dataSets.push_back(dataSet);
    return dataSet;
}

boost::shared_ptr<DataSetUser> DataManager::recordBounds(int interval, py::object collectGenerator) {
    boost::shared_ptr<DataComputer> comp = boost::shared_ptr<DataComputer>( (DataComputer *) new DataComputerBounds(state) );
    uint32_t groupTag = 1;
    boost::shared_ptr<DataSetUser> dataSet = createDataSet(comp, groupTag, interval, collectGenerator);
    dataSets.push_back(dataSet);
    return dataSet;
}

boost::shared_ptr<DataSetUser> DataManager::recordCOMV(int interval, py::object collectGenerator) {
    boost::shared_ptr<DataComputer> comp = boost::shared_ptr<DataComputer>( (DataComputer *) new DataComputerCOMV(state) );
    uint32_t groupTag = 1;
    boost::shared_ptr<DataSetUser> dataSet = createDataSet(comp, groupTag, interval, collectGenerator);
    dataSets.push_back(dataSet);
    return dataSet;
}

//computing coupling for A atoms coupling with atoms in group B
//magnetogyric ratio should be in rad/(sec*tesla)
boost::shared_ptr<MD_ENGINE::DataSetUser> DataManager::recordDipolarCoupling(std::string groupHandle, std::string groupHandleB, double magnetoA, double magnetoB, std::string computeMode, int interval, boost::python::object collectGenerator) {
    boost::shared_ptr<DataComputer> comp = boost::shared_ptr<DataComputer> ( (DataComputer *) new DataComputerDipolarCoupling(state, computeMode, groupHandle, groupHandleB, magnetoA, magnetoB));
    uint32_t groupTag = state->groupTagFromHandle(groupHandle);
    boost::shared_ptr<DataSetUser> dataSet = createDataSet(comp, groupTag, interval, collectGenerator);
    dataSets.push_back(dataSet);
    return dataSet;
}


boost::shared_ptr<MD_ENGINE::DataSetUser> DataManager::recordEField(double cutoff, int interval, boost::python::object collectGenerator) {
    boost::shared_ptr<DataComputer> comp = boost::shared_ptr<DataComputer> ( (DataComputer *) new DataComputerEField(state, cutoff));
    boost::shared_ptr<DataSetUser> dataSet = createDataSet(comp, 1, interval, collectGenerator);
    dataSets.push_back(dataSet);
    return dataSet;
}

void DataManager::addVirialTurn(int64_t t, bool perAtomVirials) {
    if (perAtomVirials) {
        clearVirialTurn(t); //to make sure there aren't two entries and that ones with perAtomVirials==true take priority
    }
    virialTurns.insert(std::make_pair(t, perAtomVirials));
}

int DataManager::getVirialModeForTurn(int64_t t) {
    auto it = virialTurns.find(std::make_pair(t, false));
    if (it != virialTurns.end()) {
        return 1;
    }
    it = virialTurns.find(std::make_pair(t, true));
    if (it != virialTurns.end()) {
        return 2;
    }
    return 0;
}

void DataManager::clearVirialTurn(int64_t t) {
    auto it = virialTurns.find(std::make_pair(t, false));
    if (it != virialTurns.end()) {
        virialTurns.erase(it);
    }
    it = virialTurns.find(std::make_pair(t, true));
    if (it != virialTurns.end()) {
        virialTurns.erase(it);
    }
}

void export_DataManager() {
    py::class_<DataManager>(
        "DataManager",
        py::no_init
    )
    .def("stopRecord", &DataManager::stopRecord)

    .def("recordTemperature", &DataManager::recordTemperature,
            (py::arg("handle") = "all",
             py::arg("mode") = "scalar",
             py::arg("interval") = 0,
             py::arg("collectGenerator") = py::object())
        )
    .def("recordEnergy", &DataManager::recordEnergy,
            (py::arg("handle") = "all",
             py::arg("mode") = "scalar",
             py::arg("interval") = 0,
             py::arg("collectGenerator") = py::object(),
             py::arg("fixes") = py::list(),
             py::arg("handleB") = "all")
        )
    .def("recordPressure", &DataManager::recordPressure,
            (py::arg("handle") = "all",
             py::arg("mode") = "scalar",
             py::arg("interval") = 0,
             py::arg("collectGenerator") = py::object())
        )
    .def("recordBounds", &DataManager::recordBounds,
            (py::arg("interval") = 0,
             py::arg("collectGenerator") = py::object())
        )
    .def("recordCOMV", &DataManager::recordCOMV,
         (py::arg("interval") = 0,
          py::arg("collectGenerator") = py::object())
        )
    .def("recordRDF", &DataManager::recordRDF,
         (py::arg("species1"),
          py::arg("species2"),
          py::arg("binWidth") = 0.01,
          py::arg("interval") = 0,
          py::arg("collectGenerator") = py::object())
        )
    .def("recordDipolarCoupling", &DataManager::recordDipolarCoupling,
            (py::arg("handleA"),
             py::arg("handleB"),
             py::arg("magnetoA"),
             py::arg("magnetoB"),
             py::arg("mode") = "scalar",
             py::arg("interval") = 0,
             py::arg("collectGenerator") = py::object())
        )
    .def("recordEField", &DataManager::recordEField,
            ( py::arg("cutoff"),
            py::arg("interval") = 0,
            py::arg("collectGenerator") = py::object())
        )
    .def("recordHamiltonian", &DataManager::recordHamiltonian,
         (py::arg("mode") = "scalar",
          py::arg("interval") = 0,
          py::arg("collectGenerator") = py::object())
        )

//boost::shared_ptr<MD_ENGINE::DataSetUser> DataManager::recordDipolarCoupling(std::string groupHandle, std::string computeMode, std::string groupHandleB, double magnetoA, double magnetoB, int interval, boost::python::object collectGenerator) {
   /* 
    .def("stopRecordBounds", &DataManager::stopRecordBounds)
    */
 //   .def("getDataSet", &DataManager::getDataSet)
    ;
}
