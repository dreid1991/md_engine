#include "DataSetBounds.h"
#include "cutils_func.h"
#include "boost_for_export.h"
#include "Bounds.h"
#include "State.h"
using namespace std;
using namespace boost::python;

DataSetBounds::DataSetBounds(State *state_, uint32_t groupTag_) : DataSet(state_, groupTag_, true, false) {
}

void computeScalar() {
    stored = state->boundsGPU;
}

void DataSetBounds::collect() {
    computeScalar();
    turns.push_back(turn);
    turnsPy.append(turn);
}
void DataSetBounds::appendValues() {
    Bounds processed = Bounds(stored);
    vals.push_back(processed);
    valsPy.append(processed);
    
}

void export_DataSetBounds() {
    class_<DataSetBounds, SHARED(DataSetBounds), bases<DataSet>, boost::noncopyable > ("DataSetBounds", no_init)
        ;
}
