#include "DataSetBounds.h"
#include "cutils_func.h"
#include "boost_for_export.h"
#include "Bounds.h"
using namespace std;
using namespace boost::python;

DataSetBounds::DataSetBounds(uint32_t groupTag_) : DataSet(groupTag_) {
}

void DataSetBounds::collect(int64_t turn, BoundsGPU &bounds, int nAtoms, float4 *xs, float4 *vs, float4 *fs, float *engs, Virial *virials, cudaDeviceProp &) {

    stored = bounds;
    turns.push_back(turn);
    turnsPy.append(turn);
}
void DataSetBounds::appendValues() {
    Bounds processed = Bounds(stored);
    vals.push_back(processed);
    valsPy.append(processed);
    //vals.push_back(Bounds(stored));
    //cout << vals.back()->lo << endl;
    
}


void export_DataSetBounds() {
    class_<DataSetBounds, SHARED(DataSetBounds), bases<DataSet>, boost::noncopyable > ("DataSetBounds", no_init)
        ;
}
