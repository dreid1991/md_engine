#include "DataSetBounds.h"
#include "cutils_func.h"
#include "boost_for_export.h"
#include "Bounds.h"
using namespace std;
using namespace boost::python;

DataSetBounds::DataSetBounds(uint32_t groupTag_) : DataSet(groupTag_) {
}

void DataSetBounds::collect(int64_t turn, BoundsGPU &bounds, int nAtoms, float4 *xs, float4 *vs, float4 *fs, float *engs, Virial *virials) {

    stored = bounds;
    turns.push_back(turn);
}
void DataSetBounds::appendValues() {
    SHARED(Bounds) toAppend = SHARED(Bounds) (new Bounds(stored));
    vals.push_back(toAppend);
    //vals.push_back(Bounds(stored));
    //cout << vals.back()->lo << endl;
    
}


void export_DataSetBounds() {
    class_<DataSetBounds, SHARED(DataSetBounds), bases<DataSet>, boost::noncopyable > ("DataSetBounds", no_init)
        .def_readwrite("vals", &DataSetBounds::vals)
        .def("getValue", &DataSetBounds::getValue)
        ;
}
