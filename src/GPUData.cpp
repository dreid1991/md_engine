#include "boost_for_export.h"
#include "GPUData.h"

unsigned int (GPUData::*switchIdx_x1) (bool) = &GPUData::switchIdx;
unsigned int (GPUData::*switchIdx_x2) ()     = &GPUData::switchIdx;

// exposing methods to python
void export_GPUData() {
    boost::python::class_<GPUData, SHARED(GPUData), boost::noncopyable> (
        "GPUData",
        boost::python::no_init
    )
    .def("activeIdx", &GPUData::activeIdx)
    .def("switchIdx", switchIdx_x1,
         boost::python::arg("onlyPositions")
         )
    .def("switchIdx", switchIdx_x2)
    ;

}
