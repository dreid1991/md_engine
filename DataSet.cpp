#include "DataSet.h"
#include "State.h"
DataSet::DataSet(State *state_, string handle_, int accumulateEvery_, int computeEvery_) : state(state_), turnInit(state->turn), handle(handle_), accumulateEvery(accumulateEvery_), computeEvery(computeEvery_) {};


void export_DataSet() {
    class_<DataSet, SHARED(DataSet) >("DataSet")
        .def_readonly("turns", &DataSet::turns)
        .def_readonly("data", &DataSet::data)
        .def_readonly("turnInit", &DataSet::turnInit)
        .def_readonly("handle", &DataSet::handle)
        .def_readwrite("computeEvery", &DataSet::computeEvery)
        ;
}
