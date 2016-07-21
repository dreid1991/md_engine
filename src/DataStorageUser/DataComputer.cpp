#include "DataComputer.h"
#include "State.h"
namespace py = boost::python;
using namespace MD_ENGINE;
DataComputer::DataComputer(State *state_, bool computingScalar_, bool computingTensor_, bool requiresEnergy_, bool requiresVirials_) {
    state = state_;
    computingScalar = computingScalar_;
    computingTensor = computingTensor_;
    requiresEnergy = requiresEnergy_;
    requiresVirials = requiresVirials_;
    lastGroupTag = 0;
};

