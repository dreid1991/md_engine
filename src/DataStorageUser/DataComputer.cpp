#include "DataComputer.h"
#include "State.h"
namespace py = boost::python;
DataComputer::DataComputer(State *state_, bool computeScalar_, bool computeTensor_) {
    state = state_;
    computeScalar = computeScalar_;
    computeTensor = computeTensor_;
    lastGroupTag = 0;
};

