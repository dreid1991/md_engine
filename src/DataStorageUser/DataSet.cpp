#include "DataSet.h"
#include "State.h"
namespace py = boost::python;
DataSet::DataSet(State *state_, uint32_t groupTag_) {
    state = state_;
    groupTag = groupTag_;

};
void export_DataSet() {
    boost::python::class_<DataSet,
                          boost::noncopyable>(
        "DataSet",
        boost::python::no_init
    )
    .def_readonly("turns", &DataSet::turnsPy)
    .def_readonly("vals", &DataSet::scalarsPy)
    .def_readonly("vectors", &DataSet::vectorsPy)
    .def_readwrite("nextCollectTurn", &DataSet::nextCollectTurn)
 //   .def("getDataSet", &DataManager::getDataSet)
    ;
}
